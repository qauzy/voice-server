package stream

import (
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"sync"
	"unicode/utf8"

	"voice_server/config"
	"voice_server/internal/logger"

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

// HandleStreamWS 处理流式 ASR WebSocket 连接，rec 为全局共享的 OnlineRecognizer
func HandleStreamWS(c *gin.Context, rec *sherpa.OnlineRecognizer) {
	if !config.GlobalConfig.StreamRecognition.Enabled {
		logger.Warnf("Stream recognition is disabled")
		c.JSON(http.StatusServiceUnavailable, map[string]string{
			"error": "stream recognition is disabled",
		})
		return
	}

	if rec == nil {
		logger.Errorf("OnlineRecognizer is nil, stream recognition unavailable")
		c.JSON(http.StatusServiceUnavailable, map[string]string{
			"error": "online recognizer not initialized",
		})
		return
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logger.Errorf("Upgrade error: %v", err)
		return
	}
	defer conn.Close()

	logger.Infof("📱 New stream connection from %s", c.Request.RemoteAddr)

	runStreamSession(conn, rec)
}

func runStreamSession(conn *websocket.Conn, rec *sherpa.OnlineRecognizer) {
	audioCh := make(chan []float32, audioChanBuf)
	var wg sync.WaitGroup

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
		processAudioStream(conn, audioCh, rec)
	}()

	wg.Wait()
	logger.Infof("🔌 Stream connection closed")
}

func processAudioStream(conn *websocket.Conn, audioCh chan []float32, rec *sherpa.OnlineRecognizer) {
	sampleRate := config.GlobalConfig.Audio.SampleRate
	if sampleRate == 0 {
		sampleRate = 16000
	}

	maxAudioLen := GetMaxAudioLen()
	if maxAudioLen == 0 {
		maxAudioLen = 20 * sampleRate
	}

	var fullText string
	var totalSamples int

	resetState := func(reason string) {
		logger.Infof("🔄 Resetting stream state: %s", reason)
		fullText = ""
		totalSamples = 0
	}

	// 每个连接独享一个 OnlineStream，不共享
	stream := CreateOnlineStream(rec)
	defer DeleteOnlineStream(stream)

	for samples := range audioCh {
		totalSamples += len(samples)
		if totalSamples > maxAudioLen {
			logger.Warnf("Audio too long, resetting stream: %d > %d", totalSamples, maxAudioLen)
			DeleteOnlineStream(stream)
			stream = CreateOnlineStream(rec)
			resetState("max_audio_len")
			continue
		}

		stream.AcceptWaveform(sampleRate, samples)

		if rec.IsReady(stream) {
			rec.Decode(stream)
		}

		result := rec.GetResult(stream)
		if result == nil {
			continue
		}

		text := result.Text
		if text == "" {
			continue
		}

		if isGarbage(text) {
			logger.Warnf("Garbage detected: %s, resetting", text)
			DeleteOnlineStream(stream)
			stream = CreateOnlineStream(rec)
			resetState("garbage")
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
			DeleteOnlineStream(stream)
			stream = CreateOnlineStream(rec)
			resetState("endpoint")
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
	n := len(data) / 4
	if n == 0 {
		return nil
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(data[i*4]) |
			uint32(data[i*4+1])<<8 |
			uint32(data[i*4+2])<<16 |
			uint32(data[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}
