package stream

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"voice-server/config"
	"voice-server/internal/logger"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

const (
	audioChanBuf = 200
)

// HandleStreamWS 处理流式 ASR WebSocket 连接
// 每个连接独立创建 OnlineRecognizer，支持 rebuildRec 彻底重建
func HandleStreamWS(c *gin.Context) {
	if !config.GlobalConfig.StreamRecognition.Enabled {
		logger.Warnf("Stream recognition is disabled")
		c.JSON(http.StatusServiceUnavailable, map[string]string{
			"error": "stream recognition is disabled",
		})
		return
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logger.Errorf("Upgrade error: %v", err)
		return
	}
	defer conn.Close()

	logger.Infof("New stream connection from %s", c.Request.RemoteAddr)

	runStreamSession(conn)
}

func runStreamSession(conn *websocket.Conn) {
	audioCh := make(chan []float32, audioChanBuf)
	var wg sync.WaitGroup

	// 每个连接独立的时间戳追踪
	var lastTextAt atomic.Int64
	var lastDecodeAt atomic.Int64
	var lastAudioAt atomic.Int64

	lastTextAt.Store(time.Now().UnixNano())
	lastDecodeAt.Store(time.Now().UnixNano())
	lastAudioAt.Store(time.Now().UnixNano())

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(audioCh)
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				logger.Warnf("Read stopped: %v", err)
				return
			}
			samples := bytesToFloat32(msg)
			if len(samples) == 0 {
				continue
			}
			select {
			case audioCh <- samples:
			default:
				logger.Warnf("Audio channel full, dropping frame")
			}
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		processAudioStream(conn, audioCh, &lastTextAt, &lastDecodeAt, &lastAudioAt)
	}()

	wg.Wait()
	logger.Infof("Stream connection closed")
}

func processAudioStream(conn *websocket.Conn, audioCh chan []float32,
	lastTextAt, lastDecodeAt, lastAudioAt *atomic.Int64) {

	sampleRate := config.GlobalConfig.Audio.SampleRate
	if sampleRate == 0 {
		sampleRate = 16000
	}

	maxAudioLen := GetMaxAudioLen()
	if maxAudioLen == 0 {
		maxAudioLen = 20 * sampleRate
	}

	// 每个连接独立创建 OnlineRecognizer
	rec := CreateOnlineRecognizer()
	stream := sherpa.NewOnlineStream(rec)

	var fullText string
	var totalSamples int

	rebuildRec := func(reason string) {
		logger.Infof("Rebuilding recognizer: %s", reason)
		sherpa.DeleteOnlineStream(stream)
		sherpa.DeleteOnlineRecognizer(rec)
		rec = CreateOnlineRecognizer()
		stream = sherpa.NewOnlineStream(rec)
		fullText = ""
		totalSamples = 0
		logger.Infof("Recognizer rebuilt ok")
	}

	defer func() {
		sherpa.DeleteOnlineStream(stream)
		sherpa.DeleteOnlineRecognizer(rec)
	}()

	for samples := range audioCh {
		lastAudioAt.Store(time.Now().UnixNano())

		// 有音频输入但超过 5s 没有任何文字输出 → 模型已失效，主动重建
		if time.Since(time.Unix(0, lastTextAt.Load())) > 5*time.Second &&
			time.Since(time.Unix(0, lastAudioAt.Load())) < time.Second {
			rebuildRec("no output timeout")
			lastTextAt.Store(time.Now().UnixNano())
			continue
		}

		totalSamples += len(samples)
		if totalSamples > maxAudioLen {
			rebuildRec("too long")
			continue
		}

		stream.AcceptWaveform(sampleRate, samples)

		if rec.IsReady(stream) {
			rec.Decode(stream)
			lastDecodeAt.Store(time.Now().UnixNano())
		}

		result := rec.GetResult(stream)
		if result == nil {
			continue
		}
		text := result.Text
		if text == "" {
			continue
		}

		lastTextAt.Store(time.Now().UnixNano())

		if isGarbage(text) {
			rebuildRec("garbage: " + text)
			continue
		}

		if text != fullText {
			incremental := strings.TrimPrefix(text, fullText)
			fullText = text
			sendJSON(conn, map[string]interface{}{
				"type": "partial",
				"text": incremental,
				"full": fullText,
			})
		}

		if rec.IsEndpoint(stream) {
			if utf8.RuneCountInString(strings.TrimSpace(fullText)) > 0 {
				sendJSON(conn, map[string]interface{}{
					"type": "final",
					"text": fullText,
				})
			}
			// endpoint 时重建 rec
			rebuildRec("final")
		}
	}
}

func sendJSON(conn *websocket.Conn, v interface{}) {
	data, _ := json.Marshal(v)
	conn.WriteMessage(websocket.TextMessage, data)
}

func isGarbage(text string) bool {
	runes := []rune(text)
	if len(runes) < 4 {
		return false
	}
	cnt := make(map[rune]int)
	for _, r := range runes {
		cnt[r]++
	}
	max := 0
	for _, v := range cnt {
		if v > max {
			max = v
		}
	}
	return float64(max)/float64(len(runes)) > 0.8
}

func bytesToFloat32(data []byte) []float32 {
	// 客户端发送 int16 PCM（每样本 2 字节，小端序）
	// 转换为 float32 归一化到 [-1.0, 1.0]
	n := len(data) / 2
	if n == 0 {
		return nil
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		// 读取 int16 小端序
		sample := int16(data[i*2]) | int16(data[i*2+1])<<8
		out[i] = float32(sample) / 32768.0
	}
	return out
}
