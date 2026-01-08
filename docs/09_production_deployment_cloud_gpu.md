# 09: Production Deployment (Cloud & GPU)

## 🚀 Moving to the Cloud

In the previous chapter (08), verify the architecture locally on our MacBook (CPU). Now, we will deploy it to a **Google Cloud L4 Instance** to unlock true production performance with **NVIDIA GPUs**.

### Why L4?
*   **Performance**: GPUs are optimized for parallel inference (matrix math).
*   **Scale**: Cloud instances can be resized or replicated.
*   **Isolation**: Keeps the heavy compute away from your laptop.

---

## ⚡ Smart Deployment (Automatic GPU Config)

We implemented a "Smart Configuration" system so you don't need to change any code between Mac and Cloud.

*   **Mac**: Uses `docker-compose.yml` (CPU layout).
*   **Cloud**: Layers `docker-compose.gpu.yml` on top (Adds NVIDIA drivers).

### Step 1: Connect to the Instance

```bash
gcloud compute ssh --zone "us-central1-c" "fmlin0429712024@isaac-sim-01" --project "prescientdemos"
```

### Step 2: Get the Latest Code

```bash
cd ~/mujoco-robot-pipeline
git pull origin main
```

### Step 3: Activate GPU Mode

Instead of editing files, we just tell Docker to use the GPU extension:

```bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
```

### Step 4: Launch!

```bash
# --build ensures we have the latest Python backend code
docker compose up -d --build
```

---

## 🔍 Verification Hierarchy

In production, "it started" isn't enough. We use a 3-tier testing strategy to guarantee reliability.

### Level 1: Smoke Test (The "Hello World")
**Goal**: Verify the server is running and reachable.
**Command**:
```bash
python scripts/test_triton_connection.py
```
**Expected**: `✓ Server is live`

### Level 2: Functional Check (10 Episodes)
**Goal**: Verify the robot logic is sound (short video).
**Command**:
```bash
export INFERENCE_MODE=triton
export TRITON_URL=localhost:8001
python scripts/eval_policy.py --mode triton --episodes 10
```

### Level 3: production Stress Test (1000 Steps)
**Goal**: Verify stability over long sessions (memory leaks, thermal throttling).
**Command**:
```bash
# This takes ~20 minutes!
python scripts/eval_policy.py --mode triton --max_steps 1000
```

---

## 🎓 Conclusion

You have successfully:
1.  **Architected** a decoupled inference system (08).
2.  **Deployed** it to a cloud GPU instance (09).
3.  **Verified** it with production-grade stress tests.

This architecture is now ready to serve real-world traffic or scaling learning fleets!
