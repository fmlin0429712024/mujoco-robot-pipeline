# Triton Python Backend Setup Guide

## Overview

This guide explains how to deploy the stateful ACT policy using Triton's Python Backend, which allows serving models that can't be exported to ONNX/TorchScript.

## Architecture

```
Container 1: Triton Server
┌─────────────────────────────────┐
│  Triton Python Backend          │
│  ├── model.py (custom code)     │
│  ├── ACT Policy (loaded)        │
│  └── GPU: L4                    │
└─────────────────────────────────┘
         ▲
         │ gRPC
         │
Container 2: Application
┌─────────────────────────────────┐
│  eval_policy.py                 │
│  ├── InferenceClient (triton)   │
│  └── No model weights!          │
└─────────────────────────────────┘
```

## Key Components

### 1. model.py (Python Backend)

Located at: `model_repository/act_pick_place/1/model.py`

**What it does:**
- Runs inside Triton container
- Loads ACT policy using PyTorch
- Handles preprocessing and inference
- Supports stateful models (action queue)

**Key methods:**
- `initialize()`: Load model once at startup
- `execute()`: Process inference requests
- `finalize()`: Cleanup on shutdown

### 2. config.pbtxt

**Changes from ONNX backend:**
```protobuf
backend: "python"  # Instead of "pytorch_libtorch"
instance_group [
  {
    kind: KIND_GPU  # Use GPU
    gpus: [ 0 ]     # L4 GPU
  }
]
```

### 3. docker-compose.yml

**Critical additions:**

**Source Code Mounting:**
```yaml
volumes:
  - ./trossen_arm_mujoco:/workspace/trossen_arm_mujoco
  - ./scripts:/workspace/scripts
  - ./outputs:/workspace/outputs
```

**Python Path:**
```yaml
environment:
  - PYTHONPATH=/workspace
```

**GPU Passthrough:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Setup Instructions

### Step 1: Verify Prerequisites

```bash
# Check GPU is available
nvidia-smi

# Check Docker has GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Step 2: Start Triton Server

```bash
# Start Triton with Python backend
docker-compose up triton
```

**Watch for these log messages:**
```
[Triton Python Backend] Loading ACT policy from: /workspace/outputs/...
[Triton Python Backend] Using GPU: NVIDIA L4
[Triton Python Backend] ✓ Model initialized successfully
```

### Step 3: Test Python Backend

```bash
# Run test script
python scripts/test_python_backend.py
```

**Expected output:**
```
✓ Server is live
✓ Server is ready
✓ Model is ready
✓ Inference successful
✓ All tests passed!
```

### Step 4: Run Inference

```bash
# Set environment
export INFERENCE_MODE=triton
export TRITON_URL=localhost:8001

# Run evaluation
python scripts/eval_policy.py --mode triton --episodes 5
```

## Troubleshooting

### Model Not Loading

**Symptom:** `Model not ready` error

**Check Triton logs:**
```bash
docker-compose logs triton | grep -A 20 "Python Backend"
```

**Common issues:**
1. **Import errors**: Source code not mounted correctly
   - Verify volumes in docker-compose.yml
   - Check PYTHONPATH is set

2. **Checkpoint not found**: Path mismatch
   - Verify CHECKPOINT_DIR environment variable
   - Check outputs/ directory is mounted

3. **GPU not available**: GPU passthrough issue
   - Verify `deploy.resources` in docker-compose.yml
   - Check `nvidia-docker` is installed

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'lerobot'`

**Solution:**
```bash
# Install dependencies in Triton container
docker-compose exec triton pip install lerobot safetensors
```

Or create a custom Triton image with dependencies pre-installed.

### GPU Not Detected

**Symptom:** Model uses CPU instead of GPU

**Check:**
```bash
# Verify GPU is passed through
docker-compose exec triton nvidia-smi
```

**Fix:**
Ensure docker-compose.yml has:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Performance Optimization

### Batch Size

Adjust in config.pbtxt:
```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 100
}
```

### Multiple Instances

For parallel requests:
```protobuf
instance_group [
  {
    count: 2  # Run 2 instances on GPU
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### Model Warmup

Add warmup to config.pbtxt:
```protobuf
model_warmup [
  {
    name: "warmup_sample"
    batch_size: 1
    inputs {
      key: "state__0"
      value: {
        data_type: TYPE_FP32
        dims: [ 8 ]
        zero_data: true
      }
    }
    inputs {
      key: "image__1"
      value: {
        data_type: TYPE_FP32
        dims: [ 3, 480, 640 ]
        zero_data: true
      }
    }
  }
]
```

## Advantages of Python Backend

1. **No Export Needed**: Serve any PyTorch model directly
2. **Stateful Models**: Handles internal state (action queue)
3. **Custom Logic**: Full Python flexibility
4. **Easy Debugging**: Standard Python debugging tools
5. **Rapid Iteration**: Change code without recompiling

## Disadvantages

1. **Performance**: Slightly slower than ONNX/TensorRT
2. **Dependencies**: Must manage Python packages
3. **GIL Limitations**: Python Global Interpreter Lock

## Next Steps

1. **Monitor Performance**: Use Prometheus metrics
2. **Optimize Batching**: Tune batch sizes for throughput
3. **Add Caching**: Cache frequent requests
4. **Scale Horizontally**: Deploy multiple Triton instances
5. **Consider TensorRT**: If export becomes possible later

## References

- [Triton Python Backend Documentation](https://github.com/triton-inference-server/python_backend)
- [Triton Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [GPU Support in Docker Compose](https://docs.docker.com/compose/gpu-support/)
