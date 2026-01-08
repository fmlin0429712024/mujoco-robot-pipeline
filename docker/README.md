# Custom Triton Docker Image Build Guide

## Overview

This directory contains the custom Dockerfile for building a Triton Inference Server image with all Python dependencies pre-installed.

## What's Included

The custom image (`Dockerfile.triton`) includes:

- **Base**: NVIDIA Triton Server 24.01-py3
- **PyTorch**: 2.1.0 (CPU version for macOS compatibility)
- **LeRobot**: Latest version with ACT policy support
- **Core Dependencies**: numpy, safetensors, pillow
- **Additional**: transformers, datasets, huggingface-hub

## Building the Image

### Automatic (via docker-compose)

```bash
# Build and start Triton
docker-compose up --build triton
```

### Manual Build

```bash
# Build the image
docker build -f docker/Dockerfile.triton -t triton-act-policy:latest ./docker

# Run the container
docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  triton-act-policy:latest
```

## Image Size Optimization

The Dockerfile uses several techniques to minimize image size:

1. **No-cache pip installs**: `--no-cache-dir` flag
2. **Cleanup**: `pip cache purge` after installations
3. **Minimal system packages**: Only essential apt packages
4. **CPU-only PyTorch**: Smaller than GPU version (for Mac)

### Expected Image Size

- Base Triton image: ~8 GB
- With dependencies: ~10-11 GB

## GPU Support (Linux Only)

To enable GPU support, modify `Dockerfile.triton`:

```dockerfile
# Replace CPU PyTorch installation with:
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

And uncomment GPU configuration in `docker-compose.yml`.

## Verification

The Dockerfile includes verification steps that run during build:

```bash
✓ PyTorch: 2.1.0
✓ LeRobot: OK
✓ SafeTensors: OK
✓ Triton Python Backend: OK
```

If the build succeeds, all dependencies are correctly installed.

## Troubleshooting

### Build Fails with "No space left on device"

**Solution**: Clean up Docker images
```bash
docker system prune -a
```

### Build is Very Slow

**Solution**: Use Docker BuildKit
```bash
DOCKER_BUILDKIT=1 docker-compose build triton
```

### Want to Add More Dependencies

Edit `Dockerfile.triton` and add to the `RUN pip install` section:

```dockerfile
RUN pip install --no-cache-dir \
    your-package-here \
    another-package
```

## Production Considerations

### Multi-stage Build (Future Optimization)

For even smaller images, consider a multi-stage build:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.10 as builder
RUN pip install --user torch lerobot

# Stage 2: Runtime
FROM nvcr.io/nvidia/tritonserver:24.01-py3
COPY --from=builder /root/.local /root/.local
```

### Caching Layers

Docker caches each layer. Order matters:
1. System packages (rarely change)
2. PyTorch (rarely change)
3. Project dependencies (may change)

This minimizes rebuild time when dependencies change.

## References

- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
