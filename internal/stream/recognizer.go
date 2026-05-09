package stream

import (
	"fmt"

	"voice-server/config"
	"voice-server/internal/logger"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func CreateOnlineRecognizer() *sherpa.OnlineRecognizer {
	cfg := config.GlobalConfig

	sampleRate := cfg.Audio.SampleRate
	if sampleRate == 0 {
		sampleRate = 16000
	}

	featureDim := cfg.Audio.FeatureDim
	if featureDim == 0 {
		featureDim = 80
	}

	streamCfg := &sherpa.OnlineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{
			SampleRate: sampleRate,
			FeatureDim: featureDim,
		},
		ModelConfig: sherpa.OnlineModelConfig{
			Transducer: sherpa.OnlineTransducerModelConfig{
				Encoder: cfg.StreamRecognition.Model.Encoder,
				Decoder: cfg.StreamRecognition.Model.Decoder,
				Joiner:  cfg.StreamRecognition.Model.Joiner,
			},
			Tokens:     cfg.StreamRecognition.Model.Tokens,
			Provider:   cfg.StreamRecognition.Provider,
			NumThreads: cfg.StreamRecognition.NumThreads,
			ModelType:  "paraformer",
		},
		DecodingMethod:          cfg.StreamRecognition.DecodingMethod,
		EnableEndpoint:          1,
		Rule1MinTrailingSilence: cfg.StreamRecognition.Endpoint.Rule1MinTrailingSilence,
		Rule2MinTrailingSilence: cfg.StreamRecognition.Endpoint.Rule2MinTrailingSilence,
		Rule3MinUtteranceLength: cfg.StreamRecognition.Endpoint.Rule3MinUtteranceLength,
	}

	rec := sherpa.NewOnlineRecognizer(streamCfg)
	if rec == nil {
		panic("failed to create online recognizer")
	}

	logger.Infof("✅ Online recognizer created: encoder=%s, decoder=%s, joiner=%s",
		cfg.StreamRecognition.Model.Encoder,
		cfg.StreamRecognition.Model.Decoder,
		cfg.StreamRecognition.Model.Joiner)

	return rec
}

func CreateOnlineStream(rec *sherpa.OnlineRecognizer) *sherpa.OnlineStream {
	if rec == nil {
		return nil
	}
	return sherpa.NewOnlineStream(rec)
}

func DeleteOnlineRecognizer(rec *sherpa.OnlineRecognizer) {
	if rec != nil {
		sherpa.DeleteOnlineRecognizer(rec)
		logger.Infof("🗑️ Online recognizer destroyed")
	}
}

func DeleteOnlineStream(stream *sherpa.OnlineStream) {
	if stream != nil {
		sherpa.DeleteOnlineStream(stream)
	}
}

func GetEndpointConfig() (rule1, rule2, rule3 float32) {
	rule1 = config.GlobalConfig.StreamRecognition.Endpoint.Rule1MinTrailingSilence
	rule2 = config.GlobalConfig.StreamRecognition.Endpoint.Rule2MinTrailingSilence
	rule3 = config.GlobalConfig.StreamRecognition.Endpoint.Rule3MinUtteranceLength

	if rule1 == 0 {
		rule1 = 2.4
	}
	if rule2 == 0 {
		rule2 = 1.2
	}
	if rule3 == 0 {
		rule3 = 20
	}

	return rule1, rule2, rule3
}

func GetMaxAudioLen() int {
	maxLen := config.GlobalConfig.StreamRecognition.MaxAudioLen
	if maxLen == 0 {
		maxLen = 20
	}
	return maxLen * config.GlobalConfig.Audio.SampleRate
}

func GetDecodingMethod() string {
	method := config.GlobalConfig.StreamRecognition.DecodingMethod
	if method == "" {
		method = "greedy_search"
	}
	return method
}

func GetEndpointEnabled() int {
	if config.GlobalConfig.StreamRecognition.Enabled {
		return 1
	}
	return 0
}

func ValidateConfig() error {
	cfg := config.GlobalConfig.StreamRecognition

	if !cfg.Enabled {
		return fmt.Errorf("stream recognition is disabled")
	}

	if cfg.Model.Encoder == "" {
		return fmt.Errorf("stream model encoder is required")
	}
	if cfg.Model.Decoder == "" {
		return fmt.Errorf("stream model decoder is required")
	}
	if cfg.Model.Joiner == "" {
		return fmt.Errorf("stream model joiner is required")
	}
	if cfg.Model.Tokens == "" {
		return fmt.Errorf("stream model tokens is required")
	}

	return nil
}
