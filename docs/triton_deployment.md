# Triton Inference Server Deployment Guide

This guide covers deploying and using the NVIDIA Triton Inference Server for ACT policy inference.

## Quick Start

### 1. Export Model to ONNX

First, export your trained ACT policy to ONNX format:

```bash
python scripts/export_model_to_triton.py \
  --checkpoint outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model \
  --output model_repository/act_pick_place/1/model.onnx
```

This will:
- Load the PyTorch checkpoint
- Export to ONNX format (or fallback to TorchScript if needed)
- Verify the exported model matches PyTorch outputs
- Place the model in the Triton repository

### 2. Start Triton Server

Using Docker Compose (recommended):

```bash
docker-compose up triton
```

Or standalone Docker:

```bash
docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

### 3. Verify Server is Running

```bash
# Check server health
curl http://localhost:8000/v2/health/ready

# Check model status
curl http://localhost:8000/v2/models/act_pick_place

# Run comprehensive test
python scripts/test_triton_connection.py
```

### 4. Run Inference

Set environment variables:

```bash
export INFERENCE_MODE=triton
export TRITON_URL=localhost:8001
```

Run evaluation:

```bash
python scripts/eval_policy.py --mode triton
```

---

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Inference mode: "triton" or "local"
INFERENCE_MODE=triton

# Triton server endpoint (gRPC)
TRITON_URL=localhost:8001

# Model configuration
MODEL_NAME=act_pick_place
MODEL_VERSION=1

# Local mode fallback
CHECKPOINT_DIR=outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model
```

### Switching Between Modes

**Triton Mode** (production):
```bash
export INFERENCE_MODE=triton
python scripts/eval_policy.py
```

**Local Mode** (development):
```bash
export INFERENCE_MODE=local
python scripts/eval_policy.py --ckpt path/to/checkpoint
```

Or use the `--mode` flag:
```bash
python scripts/eval_policy.py --mode local
python scripts/eval_policy.py --mode triton
```

---

## Model Repository Structure

```
model_repository/
└── act_pick_place/
    ├── config.pbtxt          # Triton configuration
    ├── 1/                    # Version 1
    │   └── model.onnx        # ONNX model
    └── 2/                    # Version 2 (optional)
        └── model.onnx
```

### config.pbtxt

The configuration file defines:
- **Platform**: `onnxruntime_onnx` (or `pytorch_libtorch` for TorchScript)
- **Inputs**: State (14D) and image (3x480x640)
- **Outputs**: Action (14D)
- **Batching**: Dynamic batching with preferred sizes [4, 8]
- **Instance**: CPU-based (change to `KIND_GPU` for GPU)

---

## Docker Compose Setup

The `docker-compose.yml` defines two services:

### Triton Service
- **Image**: `nvcr.io/nvidia/tritonserver:24.01-py3`
- **Ports**: 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)
- **Volume**: `./model_repository:/models`
- **Health Check**: Ensures server is ready before app starts

### App Service
- **Build**: From local Dockerfile
- **Depends On**: Triton (with health check)
- **Environment**: Configured for Triton mode
- **Volumes**: Data, outputs, visualizations

Start both services:
```bash
docker-compose up
```

Access Streamlit app at `http://localhost:8080`

---

## Performance Tuning

### Batch Size

Adjust in `config.pbtxt`:
```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]  # Increase for throughput
  max_queue_delay_microseconds: 100    # Reduce for lower latency
}
```

### Instance Count

For parallel requests:
```protobuf
instance_group [
  {
    count: 4        # Run 4 instances
    kind: KIND_CPU
  }
]
```

### GPU Acceleration

Change to GPU:
```protobuf
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]  # Use GPU 0
  }
]
```

---

## Troubleshooting

### Model Not Loading

**Symptom**: `Model not ready` error

**Solutions**:
1. Check Triton logs: `docker-compose logs triton`
2. Verify ONNX file exists: `ls model_repository/act_pick_place/1/`
3. Validate config: Check `config.pbtxt` syntax
4. Test ONNX model: `python -c "import onnx; onnx.checker.check_model('model_repository/act_pick_place/1/model.onnx')"`

### Connection Refused

**Symptom**: `Connection refused to localhost:8001`

**Solutions**:
1. Verify Triton is running: `docker ps | grep triton`
2. Check port mapping: `docker port triton-inference-server`
3. Test HTTP endpoint: `curl http://localhost:8000/v2/health/live`

### ONNX Export Fails

**Symptom**: Export script fails with ONNX errors

**Solutions**:
1. Use TorchScript fallback (automatic in export script)
2. Update `config.pbtxt` to use `pytorch_libtorch` platform
3. Install ONNX dependencies: `pip install onnx onnxruntime`

### Inference Mismatch

**Symptom**: Triton outputs differ from PyTorch

**Solutions**:
1. Check normalization: Ensure images are preprocessed correctly
2. Verify input shapes: Use `test_triton_connection.py`
3. Compare outputs: Run verification in export script

---

## Model Versioning

Deploy multiple versions:

```bash
# Export version 2
python scripts/export_model_to_triton.py \
  --checkpoint outputs/train/act_pick_place_60k/checkpoints/060000/pretrained_model \
  --output model_repository/act_pick_place/2/model.onnx

# Use version 2
export MODEL_VERSION=2
python scripts/eval_policy.py
```

Triton will load all versions. Specify version in requests or use latest.

---

## Monitoring

### Metrics Endpoint

Triton exposes Prometheus metrics at `http://localhost:8002/metrics`:

```bash
curl http://localhost:8002/metrics | grep nv_inference
```

Key metrics:
- `nv_inference_request_success`: Successful requests
- `nv_inference_request_failure`: Failed requests
- `nv_inference_queue_duration_us`: Queue time
- `nv_inference_compute_infer_duration_us`: Inference time

### Logging

View detailed logs:
```bash
docker-compose logs -f triton
```

Enable verbose logging in `docker-compose.yml`:
```yaml
command: tritonserver --model-repository=/models --log-verbose=1
```

---

## Production Deployment

### Kubernetes

Example deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        args: ["tritonserver", "--model-repository=/models"]
        volumeMounts:
        - name: model-repo
          mountPath: /models
        resources:
          limits:
            nvidia.com/gpu: 1
      volumes:
      - name: model-repo
        persistentVolumeClaim:
          claimName: model-pvc
```

### Cloud Deployment

- **AWS**: Use ECS/EKS with GPU instances
- **GCP**: Use GKE with GPU node pools
- **Azure**: Use AKS with GPU VMs

Mount model repository from cloud storage (S3, GCS, Azure Blob).

---

## Next Steps

1. **Optimize Model**: Use TensorRT for faster inference
2. **Scale Horizontally**: Deploy multiple Triton instances
3. **Add Monitoring**: Integrate with Prometheus/Grafana
4. **Implement A/B Testing**: Route traffic between model versions
5. **Enable Caching**: Cache frequent requests

For more details, see [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/).
