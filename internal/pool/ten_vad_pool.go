package pool

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"voice-server/internal/logger"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// SherpaTenVADConfig Sherpa TEN-VAD配置（使用 sherpa-onnx 内置 API）
type SherpaTenVADConfig struct {
	ModelConfig       *sherpa.VadModelConfig
	BufferSizeSeconds float32
	PoolSize          int
	MaxIdle           int
}

// SherpaTenVADInstance Sherpa TEN-VAD实例
type SherpaTenVADInstance struct {
	ID       int
	VAD      *sherpa.VoiceActivityDetector
	LastUsed int64
	InUse    int32
	mu       sync.RWMutex
}

// GetID 获取实例ID
func (i *SherpaTenVADInstance) GetID() int {
	return i.ID
}

// GetType 获取VAD类型
func (i *SherpaTenVADInstance) GetType() string {
	return TEN_VAD_TYPE
}

// IsInUse 检查是否在使用中
func (i *SherpaTenVADInstance) IsInUse() bool {
	return atomic.LoadInt32(&i.InUse) == 1
}

// SetInUse 设置使用状态
func (i *SherpaTenVADInstance) SetInUse(inUse bool) {
	if inUse {
		atomic.StoreInt32(&i.InUse, 1)
	} else {
		atomic.StoreInt32(&i.InUse, 0)
	}
}

// GetLastUsed 获取最后使用时间
func (i *SherpaTenVADInstance) GetLastUsed() int64 {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.LastUsed
}

// SetLastUsed 设置最后使用时间
func (i *SherpaTenVADInstance) SetLastUsed(timestamp int64) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.LastUsed = timestamp
}

// Reset 重置实例状态
func (i *SherpaTenVADInstance) Reset() error {
	if i.VAD != nil {
		for !i.VAD.IsEmpty() {
			segment := i.VAD.Front()
			i.VAD.Pop()
			if segment != nil {
			}
		}
	}
	return nil
}

// Destroy 销毁实例
func (i *SherpaTenVADInstance) Destroy() error {
	if i.VAD != nil {
		sherpa.DeleteVoiceActivityDetector(i.VAD)
		i.VAD = nil
		logger.Infof("🗑️ Sherpa TEN-VAD instance destroyed")
	}
	return nil
}

// SherpaTenVADPool Sherpa TEN-VAD资源池
type SherpaTenVADPool struct {
	instances []*SherpaTenVADInstance
	available chan VADInstanceInterface
	config    *SherpaTenVADConfig

	// 统计信息
	totalCreated int64
	totalReused  int64
	totalActive  int64

	// 控制
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

// NewSherpaTenVADPool 创建新的Sherpa TEN-VAD资源池
func NewSherpaTenVADPool(config *SherpaTenVADConfig) *SherpaTenVADPool {
	ctx, cancel := context.WithCancel(context.Background())

	pool := &SherpaTenVADPool{
		instances: make([]*SherpaTenVADInstance, 0, config.PoolSize),
		available: make(chan VADInstanceInterface, config.PoolSize),
		config:    config,
		ctx:       ctx,
		cancel:    cancel,
	}

	return pool
}

// Initialize 并行初始化VAD池
func (p *SherpaTenVADPool) Initialize() error {
	logger.Infof("🔧 Initializing Sherpa TEN-VAD pool with %d instances...", p.config.PoolSize)

	var initWg sync.WaitGroup
	errorChan := make(chan error, p.config.PoolSize)

	for i := 0; i < p.config.PoolSize; i++ {
		initWg.Add(1)
		go func(instanceID int) {
			defer initWg.Done()

			vad := sherpa.NewVoiceActivityDetector(p.config.ModelConfig, p.config.BufferSizeSeconds)
			if vad == nil {
				errorChan <- fmt.Errorf("failed to create Sherpa TEN-VAD instance %d", instanceID)
				return
			}

			instance := &SherpaTenVADInstance{
				VAD:      vad,
				LastUsed: time.Now().UnixNano(),
				InUse:    0,
				ID:       instanceID,
			}

			p.mu.Lock()
			p.instances = append(p.instances, instance)
			p.mu.Unlock()

			select {
			case p.available <- instance:
				atomic.AddInt64(&p.totalCreated, 1)
				logger.Infof("✅ Sherpa TEN-VAD instance %d initialized", instanceID)
			default:
				sherpa.DeleteVoiceActivityDetector(vad)
				errorChan <- fmt.Errorf("Sherpa TEN-VAD pool queue full, instance %d discarded", instanceID)
			}
		}(i)
	}

	initWg.Wait()
	close(errorChan)

	var initErrors []error
	for err := range errorChan {
		if err != nil {
			initErrors = append(initErrors, err)
			logger.Warnf("⚠️ Sherpa TEN-VAD initialization warning: %v", err)
		}
	}

	successCount := len(p.instances)
	logger.Infof("🚀 Sherpa TEN-VAD pool initialized with %d/%d instances", successCount, p.config.PoolSize)

	if len(initErrors) > 0 && successCount == 0 {
		return fmt.Errorf("failed to initialize any Sherpa TEN-VAD instances")
	}

	return nil
}

// Get 获取VAD实例
func (p *SherpaTenVADPool) Get() (VADInstanceInterface, error) {
	logger.Infof("🔍 Attempting to get Sherpa TEN-VAD instance from pool (available: %d)", len(p.available))

	select {
	case instance := <-p.available:
		logger.Infof("🎯 Got Sherpa TEN-VAD instance %d from pool", instance.GetID())
		if atomic.CompareAndSwapInt32(&instance.(*SherpaTenVADInstance).InUse, 0, 1) {
			instance.SetLastUsed(time.Now().UnixNano())
			atomic.AddInt64(&p.totalReused, 1)
			atomic.AddInt64(&p.totalActive, 1)
			logger.Infof("✅ Sherpa TEN-VAD instance %d marked as in-use (active: %d)", instance.GetID(), atomic.LoadInt64(&p.totalActive))
			return instance, nil
		}
		logger.Warnf("⚠️ Sherpa TEN-VAD instance %d already in use, returning to pool", instance.GetID())
		select {
		case p.available <- instance:
		default:
		}
		return p.Get()
	case <-time.After(100 * time.Millisecond):
		logger.Warnf("⏰ Sherpa TEN-VAD pool timeout, creating new temporary instance")
		return p.createNewInstance()
	case <-p.ctx.Done():
		logger.Errorf("❌ Sherpa TEN-VAD pool is shutting down")
		return nil, fmt.Errorf("Sherpa TEN-VAD pool is shutting down")
	}
}

// Put 归还VAD实例
func (p *SherpaTenVADPool) Put(instance VADInstanceInterface) {
	if instance == nil {
		logger.Warnf("⚠️ Attempted to put nil Sherpa TEN-VAD instance")
		return
	}

	logger.Infof("🔄 Returning Sherpa TEN-VAD instance %d to pool", instance.GetID())

	if atomic.CompareAndSwapInt32(&instance.(*SherpaTenVADInstance).InUse, 1, 0) {
		instance.SetLastUsed(time.Now().UnixNano())
		atomic.AddInt64(&p.totalActive, -1)
		logger.Infof("✅ Sherpa TEN-VAD instance %d marked as available (active: %d)", instance.GetID(), atomic.LoadInt64(&p.totalActive))

		if err := instance.Reset(); err != nil {
			logger.Warnf("⚠️ Failed to reset Sherpa TEN-VAD instance %d: %v", instance.GetID(), err)
		}

		select {
		case p.available <- instance:
			logger.Infof("✅ Sherpa TEN-VAD instance %d returned to pool (available: %d)", instance.GetID(), len(p.available))
		default:
			logger.Warnf("⚠️ Sherpa TEN-VAD pool queue full, destroying instance %d", instance.GetID())
			instance.Destroy()
		}
	} else {
		logger.Warnf("⚠️ Sherpa TEN-VAD instance %d was not in use, cannot return", instance.GetID())
	}
}

// createNewInstance 创建新的VAD实例
func (p *SherpaTenVADPool) createNewInstance() (VADInstanceInterface, error) {
	vad := sherpa.NewVoiceActivityDetector(p.config.ModelConfig, p.config.BufferSizeSeconds)
	if vad == nil {
		return nil, fmt.Errorf("failed to create new Sherpa TEN-VAD instance")
	}

	instance := &SherpaTenVADInstance{
		VAD:      vad,
		LastUsed: time.Now().UnixNano(),
		InUse:    1,
		ID:       -1,
	}

	atomic.AddInt64(&p.totalCreated, 1)
	atomic.AddInt64(&p.totalActive, 1)

	logger.Infof("🆕 Created temporary Sherpa TEN-VAD instance")
	return instance, nil
}

// GetStats 获取统计信息
func (p *SherpaTenVADPool) GetStats() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return map[string]interface{}{
		"vad_type":        TEN_VAD_TYPE,
		"pool_size":       p.config.PoolSize,
		"max_idle":        p.config.MaxIdle,
		"total_instances": len(p.instances),
		"available_count": len(p.available),
		"active_count":    atomic.LoadInt64(&p.totalActive),
		"total_created":   atomic.LoadInt64(&p.totalCreated),
		"total_reused":    atomic.LoadInt64(&p.totalReused),
	}
}

// Shutdown 关闭VAD池
func (p *SherpaTenVADPool) Shutdown() {
	logger.Infof("🛑 Shutting down Sherpa TEN-VAD pool...")

	p.cancel()

	p.mu.Lock()
	defer p.mu.Unlock()

	for {
		select {
		case instance := <-p.available:
			instance.Destroy()
		default:
			goto cleanup_instances
		}
	}

cleanup_instances:
	for _, instance := range p.instances {
		instance.Destroy()
	}

	p.instances = nil
	close(p.available)

	logger.Infof("✅ Sherpa TEN-VAD pool shutdown complete")
}
