# Deployment Guide for GCP L4 Instance

## Quick Start on L4 Instance

### 1. SSH into Your VM

```bash
gcloud compute ssh --zone "us-central1-c" "isaac-sim-01" --project "prescientdemos"
```

### 2. Clone/Pull the Repository

```bash
# If first time
git clone https://github.com/fmlin0429712024/mujoco-robot-pipeline.git
cd mujoco-robot-pipeline

# If already cloned
cd mujoco-robot-pipeline
git pull origin main
```

### 3. Verify Files Are Present

```bash
ls -la docker/Dockerfile.triton
ls -la docker-compose.yml
ls -la model_repository/act_pick_place/1/model.py
ls -la model_repository/act_pick_place/config.pbtxt
```

**Expected output:**
```
✓ docker/Dockerfile.triton
✓ docker-compose.yml  
✓ model_repository/act_pick_place/1/model.py
✓ model_repository/act_pick_place/config.pbtxt
```

### 4. Install Docker (if needed)

```bash
# Check if Docker is installed
docker --version

# If not installed:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker  # Refresh group membership
```

### 5. Install NVIDIA Container Toolkit (for GPU)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 6. Enable GPU in docker-compose.yml

Uncomment the GPU configuration:

```yaml
# In docker-compose.yml, uncomment:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 7. Update Dockerfile for GPU PyTorch

Edit `docker/Dockerfile.triton`:

```dockerfile
# Replace CPU PyTorch with GPU version:
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 8. Build the Custom Triton Image

```bash
# This will be MUCH faster on L4 than on Mac!
docker-compose build triton

# Expected time: 5-10 minutes (vs 30+ minutes on Mac)
```

### 9. Copy Model Checkpoint to VM

You need to transfer your trained model from Mac to L4:

**Option A: Using gcloud scp**
```bash
# On your Mac:
gcloud compute scp --recurse \
  outputs/train/act_pick_place_30k \
  isaac-sim-01:~/mujoco-robot-pipeline/outputs/train/ \
  --zone "us-central1-c" \
  --project "prescientdemos"
```

**Option B: Using Google Cloud Storage**
```bash
# On Mac: Upload to GCS
gsutil -m cp -r outputs/train/act_pick_place_30k gs://your-bucket/

# On L4: Download from GCS
gsutil -m cp -r gs://your-bucket/act_pick_place_30k outputs/train/
```

### 10. Start Triton Server

```bash
docker-compose up triton
```

**Watch for:**
```
[Triton Python Backend] Loading ACT policy from: /workspace/outputs/...
[Triton Python Backend] Using GPU: NVIDIA L4
[Triton Python Backend] ✓ Model initialized successfully
```

### 11. Test the Deployment

In another terminal on the L4 instance:

```bash
# Install tritonclient
pip install tritonclient[grpc]

# Run test
python scripts/test_python_backend.py
```

**Expected output:**
```
✓ Server is live
✓ Server is ready  
✓ Model is ready
✓ Inference successful
✓ Latency: ~50ms (much faster with GPU!)
```

### 12. Run Evaluation

```bash
export INFERENCE_MODE=triton
export TRITON_URL=localhost:8001

python scripts/eval_policy.py --mode triton --episodes 10
```

## Troubleshooting on L4

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Model Not Found

```bash
# Verify checkpoint exists
ls -la outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model/

# Check Triton logs
docker-compose logs triton | grep "Loading ACT policy"
```

### Build Fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache triton
```

## Performance Comparison

| Metric | Mac (M1/M2) | L4 Instance |
|--------|-------------|-------------|
| **Docker Build** | 30+ min | 5-10 min |
| **Inference (CPU)** | ~200ms | ~100ms |
| **Inference (GPU)** | N/A | ~50ms |
| **Architecture** | ARM64 (emulated) | x86_64 (native) |

## Next Steps

1. ✅ Build completes successfully
2. ✅ Model loads with GPU
3. ✅ Test inference works
4. ✅ Run full evaluation
5. 🎯 Deploy to production!

## Files Checklist

All these files are now on GitHub (commit `4f0eaa7`):

- ✅ `docker/Dockerfile.triton` - Custom Triton image
- ✅ `docker-compose.yml` - Build configuration
- ✅ `model_repository/act_pick_place/1/model.py` - Python backend
- ✅ `model_repository/act_pick_place/config.pbtxt` - Triton config
- ✅ `scripts/test_python_backend.py` - Test script
- ✅ `PYTHON_BACKEND_SETUP.md` - Detailed guide

**You're all set to pull and build on your L4 instance!** 🚀
