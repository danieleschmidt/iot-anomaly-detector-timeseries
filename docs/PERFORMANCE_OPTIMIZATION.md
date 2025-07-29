# Performance Optimization Guide

This document outlines advanced performance optimization strategies for production deployments of the IoT Anomaly Detection system.

## Performance Profiling

### CPU Profiling with py-spy
```bash
# Profile the training process
py-spy record -o profile.svg -- python -m src.train_autoencoder --epochs 10

# Profile API server under load
py-spy record -o api_profile.svg --pid $(pgrep -f model_serving_api)
```

### Memory Profiling with memray
```bash
# Profile memory usage during training
memray run --output training_memory.bin python -m src.train_autoencoder
memray flamegraph training_memory.bin

# Profile API server memory
memray run --output api_memory.bin python -m src.model_serving_api
```

## Optimization Strategies

### Model Optimization
- **Quantization**: Reduce model size by 75% with minimal accuracy loss
- **Pruning**: Remove redundant parameters for faster inference
- **ONNX Conversion**: Deploy optimized models with ONNX Runtime

### Data Pipeline Optimization
- **Batched Processing**: Process multiple samples simultaneously
- **Async I/O**: Non-blocking data loading and preprocessing
- **Memory Mapping**: Efficient large dataset handling

### Infrastructure Optimization
- **Container Resource Limits**: Right-size CPU/memory allocation  
- **Auto-scaling**: Dynamic resource adjustment based on load
- **CDN Integration**: Distribute static assets globally

## Monitoring and Alerting

Performance metrics are automatically collected via Prometheus and visualized in Grafana dashboards at `config/grafana-dashboard.json`.

Key metrics to monitor:
- Inference latency (p95 < 100ms target)
- Memory usage (< 2GB per container)
- CPU utilization (< 80% sustained)
- Error rates (< 0.1% target)