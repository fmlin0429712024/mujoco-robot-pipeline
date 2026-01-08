# Triton Inference Server Deployment Guide

This guide covers deploying and using the NVIDIA Triton Inference Server for ACT policy inference.

> [!IMPORTANT]
> **Current Status**: The inference abstraction layer is fully implemented and working in **local mode**. Triton deployment is ready but model export is blocked by ACT model complexity (internal state makes ONNX/TorchScript export challenging). See [Export Challenges](#export-challenges) section below.

## Quick Start (Local Mode - Working Now)

### 1. Run Inference Locally

The easiest way to use the new inference architecture:

```bash
# Set mode to local (or omit, it's the default)
export INFERENCE_MODE=local

# Run evaluation
python scripts/eval_policy.py --mode local --episodes 10
```

This uses the `InferenceClient` abstraction with local PyTorch inference - **fully functional and tested**.

### 2. Test the Inference Client

Debug and verify the setup:

```bash
python scripts/debug_policy_api.py
```

This will:
- Load the policy from checkpoint
- Initialize the inference client
- Run a test inference with dummy data
- Report success/failure

---

## Triton Mode (Infrastructure Ready, Export Blocked)

The Triton infrastructure is fully implemented but requires model export, which is currently challenging for ACT models.

### Why Export is Challenging

The ACT policy has **internal state** that makes standard export difficult:

```python
# Inside ACT model
self._action_queue = []  # Maintains history between calls
```

**Issues encountered:**
- ✗ ONNX export: Requires `action` key in forward() that's not available during inference
- ✗ TorchScript tracing: Internal state prevents tracing
- ✗ TorchScript scripting: PyTorch internal assertion failures

### If You Want to Try Triton (Advanced)

**Option 1: Use PyTorch Backend** (attempted, but has issues)
```bash
python scripts/save_model_for_triton.py
```

**Option 2: Wait for LeRobot Export Support**
The LeRobot team may add official export support in future versions.

**Option 3: Use a Simpler Model**
If you train a different policy without internal state, the export scripts will work.


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
    └── 1/                    # Version 1
        └── model.pt          # PyTorch model (when export works)
```

### config.pbtxt

The configuration file defines:
- **Platform**: `pytorch_libtorch` (for PyTorch models)
- **Inputs**: State (8D) and image (3x480x640)
- **Outputs**: Action (8D)
- **Batching**: Dynamic batching with preferred sizes [4, 8]
- **Instance**: CPU-based (change to `KIND_GPU` for GPU)

**Note**: Input/output names for PyTorch backend use positional notation (`state__0`, `image__1`, `output__0`).

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

## Export Challenges

### ACT Model Complexity

The ACT policy from LeRobot has characteristics that make standard export difficult:

**1. Stateful Architecture**
```python
class ACTPolicy:
    def __init__(self):
        self._action_queue = []  # Internal state
```

The model maintains an action queue between inference calls, which violates the stateless assumption of ONNX/TorchScript.

**2. Forward Method Signature**
```python
def forward(self, batch):
    # Expects 'action' key for training
    loss = F.l1_loss(batch["action"], predicted_action)
```

The `forward()` method requires an `action` key that's not available during inference, making direct export impossible.

**3. Complex Control Flow**
The model uses dynamic operations (list extensions, conditional logic) that are hard to trace.

### Attempted Solutions

**ONNX Export**: Failed due to missing `action` key in batch
**TorchScript Tracing**: Failed due to internal state modifications
**TorchScript Scripting**: Failed with PyTorch internal assertion errors
**Wrapper Approach**: Created inference wrapper, but still hits export limitations

### Workarounds

**Current**: Use local mode with `InferenceClient` - works perfectly!
**Future**: Wait for LeRobot to add official export support
**Alternative**: Use Triton's Python backend (custom code, not standard model serving)

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
