# System Architecture Overview

> **Quick Reference:** Understand how the robot learning pipeline works in 5 minutes

---

## 🎯 What This System Does

Teaches a 6-DOF robot arm to **pick up a red cube and place it in a green bucket** using:
1. 50 expert demonstrations (scripted policy)
2. Vision-based imitation learning (ACT policy)
3. Simulation-only deployment

---

## 📐 Architecture (4 Layers)

```
┌─────────────────────────────────────────────────────────────┐
│                    ROBOT LEARNING PIPELINE                   │
└─────────────────────────────────────────────────────────────┘

Layer 1: SIMULATION          Layer 2: DATA           Layer 3: LEARNING
┌──────────────────┐         ┌──────────────┐        ┌──────────────┐
│ trossen_arm_     │         │ scripts/     │        │ scripts/     │
│ mujoco/          │────────▶│              │───────▶│              │
│                  │         │              │        │              │
│ • Scene (XML)    │  HDF5   │ • Convert    │ Parquet│ • Train      │
│ • Expert policy  │  26GB   │ • Visualize  │ 447MB  │ • Evaluate   │
│ • Record script  │         │              │        │              │
└──────────────────┘         └──────────────┘        └──────────────┘
```

---

## 📂 What Each Directory Does

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| **`trossen_arm_mujoco/`** | Simulation core | `sim_env.py` (task), `scripted_policy.py` (expert), `record_sim_episodes.py` (data collection) |
| **`scripts/`** | Data + Learning pipeline | `create_lerobot_dataset.py` (convert), `train_policy.py` (train), `eval_policy.py` (evaluate) |
| **`data/lerobot/`** | Training dataset | Parquet tables + MP4 videos (447MB) |
| **`outputs/train/`** | Trained models | Checkpoints (591MB) |
| **`visualizations/`** | Demo videos | Expert, random, before/after training |

---

## 🔄 Complete Data Flow

```
1. Expert Demos
   ├─▶ record_sim_episodes.py
   └─▶ data/raw/*.hdf5 (26GB)

2. Convert Format
   ├─▶ create_lerobot_dataset.py
   └─▶ data/lerobot/ (447MB)            ← Delete data/raw after this!

3. Train Policy
   ├─▶ train_policy.py (30k steps)
   └─▶ outputs/train/.../030000/

4. Evaluate
   ├─▶ eval_policy.py
   └─▶ visualizations/after_training.mp4
```

---

## 🎓 Training-Eval Loop Explained

### How Training Works

**Simple Answer:** Train for a **fixed number of steps** (30,000), then stop and evaluate.

**Not like this** ❌:
```
Train 1000 steps → Eval → Check if perfect → If not, train more → Repeat
```

**Actually like this** ✅:
```
Train continuously for 30,000 steps → Stop → Eval once → Done
```

### Why This Design?

| Design Choice | Reason |
|---------------|--------|
| **Fixed 30k steps** | Simpler, more predictable than convergence-based stopping |
| **No periodic eval** | Saves time, eval is expensive (run full episodes) |
| **Eval at end only** | You decide if you want to train longer after seeing results |

### Detailed Timeline

```
Step 0        Step 5k       Step 10k      Step 15k      Step 20k      Step 25k      Step 30k
│             │             │             │             │             │             │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Training... │ Save model  │ Save model  │ Save model  │ Save model  │ Save model  │ STOP
│             │             │             │             │             │             │
│             checkpoint    checkpoint    checkpoint    checkpoint    checkpoint    ▼
│                                                                                    Eval
│                                                                                    (10 episodes)
│                                                                                    │
Loss: 4.374                                                              0.036     Success: 0%
```

**What happens:**
1. **Training phase** (0-30k steps):
   - Read random batches from dataset
   - Compute loss (how different is policy from expert?)
   - Update model weights
   - Save checkpoint every 5k steps
   - **No evaluation during training**

2. **After training** (manual):
   - Run `eval_policy.py` on final checkpoint
   - Simulate 10 episodes, measure success rate
   - Record video to see what policy learned
   - **You decide:** Train longer? Collect more data? Ship it?

### When to Stop Training?

**Option A: Fixed Budget** (This project ✅)
- Decide upfront: "Train for 30k steps"
- Eval once at end
- Simple, predictable

**Option B: Convergence-Based** (Not used here)
- Monitor validation loss during training
- Stop when loss plateaus
- More complex, needs validation set

**Why we chose Option A:**
- Simpler for learning project
- 30k steps = ~2-3 hours on Apple Silicon
- Can always train longer if needed

---

## 🔧 Critical Components Explained

### 1. Action Recording (Most Important!)

**The Bug We Fixed:**
```python
# ❌ WRONG (identity mapping)
action[t] = current_position

# ✅ CORRECT (next position)
action[t] = next_position
```

**Why this matters:** Policy needs to learn "given current state, what's the next state?" not "output what you see."

### 2. Image Preprocessing

**Training and eval MUST use the same normalization:**
```python
# Both must apply ImageNet normalization
image = (image / 255.0 - mean) / std
```

**Why this matters:** Policy trained on normalized images will fail if eval gives unnormalized images.

### 3. Data Format Conversion

**HDF5 → LeRobot Parquet:**
- **Why convert?** Parquet is faster for training (column-oriented)
- **Compression:** 26GB → 447MB (58x smaller!)
- **Videos:** MP4 encoding instead of raw frames

---

## 💡 Quick Reference: Common Tasks

### Task: Retrain from scratch
```bash
# Keep dataset, delete models
rm -rf outputs/train/*
python scripts/train_policy.py
```

### Task: Train longer (extend 30k → 50k)
```bash
# Modify train_policy.py: steps = 50000
python scripts/train_policy.py
```

### Task: Test different checkpoint
```bash
python scripts/eval_policy.py --ckpt outputs/train/.../020000/pretrained_model
```

### Task: Free up disk space
```bash
# After conversion, delete raw data
rm -rf data/raw/*

# After training, delete intermediate checkpoints
rm -rf outputs/train/.../checkpoints/{005000,010000,015000,020000,025000}
```

---

## 🎯 Success Criteria

### What "Success" Looks Like

| Metric | Target | This Project |
|--------|--------|--------------|
| Training loss | < 0.05 | ✅ 0.036 |
| Eval success rate | > 80% | ❌ 0% |
| Arm reaches cube | Yes | ✅ Yes (shows intent) |
| Arm grasps cube | Yes | ❌ Not yet |

**Interpretation:** Policy learned movement toward cube but needs more training/data for precision.

---

## 🚀 Next Steps to Improve

**If you want better success rate:**
1. Record more data (100+ episodes instead of 50)
2. Train longer (50k steps instead of 30k)
3. Add data augmentation (vary lighting/colors)

**If you want real robot deployment:**
1. Migrate to Isaac Sim (photorealistic)
2. Add domain randomization
3. Acquire physical Trossen arm
4. See [MuJoCo vs Isaac Sim comparison](mujoco_vs_isaac_sim.md)

---

## 📚 Key Takeaways

✅ **Architecture:** 4 layers (Simulation → Data → Learning → Storage)

✅ **Training:** Fixed 30k steps, no periodic evaluation, eval once at end

✅ **Data:** HDF5 for recording → Parquet for training (compression + speed)

✅ **Critical:** Action recording bug fix, image normalization consistency

✅ **Results:** Loss decreased (4.3→0.04) but 0% success (needs more data/training)

**This is a complete, working learning pipeline** - you can now understand every component and how they connect! 🎉
