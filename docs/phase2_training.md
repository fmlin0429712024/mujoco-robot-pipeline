# Phase 2: Training ACT Policy

## Objective
Train the ACT (Action Chunking with Transformers) policy for 30,000 steps and generate baseline/comparison videos.

## Prerequisites

✅ Phase 1 completed (50 episodes in `data/lerobot/sim_pick_place_demo/`)

## Steps

### 1. Generate Baseline Videos

Before training, create comparison videos to show the difference:

#### Random Policy Baseline
```bash
python scripts/visualize_random_policy.py
```

**Expected output:**
- `visualizations/random_policy.mp4` (~1.5MB)
- Shows random, uncoordinated arm movements

#### Untrained Policy
```bash
python scripts/visualize_untrained_policy.py
```

**Expected output:**
- `visualizations/untrained_policy.mp4` (~0.7MB)
- Shows minimal/no movement (policy at initialization)

**⏱️ Time:** ~2 minutes total

---

### 2. Train ACT Policy

Train for 30,000 steps on the collected dataset:

```bash
python scripts/train_policy.py 2>&1 | tee training.log
```

**Training configuration:**
- Policy: ACT (Action Chunking Transformers)
- Steps: 30,000
- Batch size: 8
- Device: MPS (Apple Silicon) or CUDA (NVIDIA GPU)
- Expected loss: 4.374 → ~0.036

**Expected output:**
- Checkpoints saved every 5,000 steps in `outputs/train/act_pick_place_30k/checkpoints/`
- Log showing decreasing loss over time
- Final checkpoint at `outputs/train/act_pick_place_30k/checkpoints/030000/`

**⏱️ Time:** ~2-3 hours on Apple Silicon M-series

---

### 3. Monitor Training Progress

Watch the training log in real-time:

```bash
tail -f training.log
```

**Key milestones:**
```
Step 5k:  Loss ~0.073
Step 10k: Loss ~0.057
Step 20k: Loss ~0.043
Step 30k: Loss ~0.036
```

Press `Ctrl+C` to stop viewing the log.

---

## Verification

Check that training completed successfully:

```bash
# Verify final checkpoint exists
ls -lh outputs/train/act_pick_place_30k/checkpoints/030000/

# Check training log
grep "step:30K" training.log

# Check disk usage
du -sh outputs/train/act_pick_place_30k/
```

**Expected checkpoint size:** ~591 MB

## Next Steps

✅ Phase 2 complete! Proceed to [Phase 3: Deployment](phase3_deployment.md)
