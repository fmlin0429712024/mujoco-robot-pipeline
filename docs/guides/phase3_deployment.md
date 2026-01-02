# Phase 3: Deploy Trained Model

## Objective
Deploy the trained ACT policy in simulation and record the deployment video showing the learned behavior.

## Prerequisites

✅ Phase 2 completed (trained model at `outputs/train/act_pick_place_30k/checkpoints/030000/`)

## Steps

### 1. Evaluate Trained Policy

Run the trained policy for 10 episodes and generate a video:

```bash
python scripts/eval_policy.py \
  --ckpt outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model
```

**What this does:**
- Loads the trained 30k checkpoint
- Runs 10 evaluation episodes with different random seeds
- Records all episodes into a single video
- Reports success rate

**Expected output:**
- `visualizations/after_training.mp4` (~23MB)
- Console output showing episode results (success/failure per episode)
- Success rate: 0-100% (this project achieved 0% but shows learning intent)

**⏱️ Time:** ~5-10 minutes

---

### 2. Analyze Results

Watch the deployment video to see what the policy learned:

```bash
open visualizations/after_training.mp4  # macOS
# or
vlc visualizations/after_training.mp4   # Linux
```

**What to look for:**
- Does the arm move toward the cube? ✓
- Does it attempt to grasp? ✓
- Does it complete the task? (May vary)
- Compare with `random_policy.mp4` baseline

---

### 3. Compare All Videos

You now have all 4 videos showing the complete learning pipeline:

| Video | Shows |
|-------|-------|
| `expert_demo.mp4` | Expert policy (ground truth) |
| `random_policy.mp4` | Random baseline (no learning) |
| `untrained_policy.mp4` | Before training |
| `after_training.mp4` | **After training (your result!)** |

---

## Verification

```bash
# List all generated videos
ls -lh visualizations/*.mp4

# Check video sizes
du -sh visualizations/
```

**Expected videos:**
- expert_demo.mp4 (~2MB)
- random_policy.mp4 (~1.5MB)
- untrained_policy.mp4 (~0.7MB)
- after_training.mp4 (~23MB)

---

## Upload to YouTube (Optional)

1. Upload all 4 videos to YouTube
2. Update `README.md` with your video links:
   ```markdown
   | **Phase 1** | [Expert Demo](YOUR_YOUTUBE_LINK) |
   | **Phase 1** | [Random Policy](YOUR_YOUTUBE_LINK) |
   | **Phase 1** | [Before Training](YOUR_YOUTUBE_LINK) |
   | **Phase 2** | [After Training](YOUR_YOUTUBE_LINK) |
   ```

---

## Final Steps

✅ Phase 3 complete! Now proceed to [Cleanup](cleanup.md) to free disk space.
