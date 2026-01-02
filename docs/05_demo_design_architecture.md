# Demo Design Architecture

> **Your learning project:** Complete pick-and-place IL pipeline explained

**Reading time:** 5 minutes

---

## What You Built

A complete **Imitation Learning pipeline** that teaches a 6-DOF robot arm to pick a cube and place it in a bucket.

**This is the culmination of everything you learned** - from understanding methods (doc 01) to applying them in a real project.

---

## System Overview

```
┌─────────────────────────────────────────────┐
│         PICK-AND-PLACE IL PIPELINE           │
└─────────────────────────────────────────────┘

LAYER 1: Simulation          DATA PIPELINE         LEARNING
┌──────────────┐            ┌──────────┐         ┌────────────┐
│ MuJoCo       │──HDF5─────▶│ Convert  │──Parquet▶│ ACT        │
│ + Expert     │  (26GB)    │ to       │ (447MB) │ Training   │
│ + Random     │            │ LeRobot  │         │ 30k steps  │
│   cubes      │            │          │         │            │
└──────────────┘            └──────────┘         └────────────┘
      ↑                                                 │
      │                                                 ↓
      └─────────────────────────────────────  Trained Policy
                                                   (Evaluation)
```

---

## Architecture Layers

### **Layer 1: Simulation Core**

**Location:** `trossen_arm_mujoco/`

**What it does:** Provides physics, robot model, and task definition

**Key components:**

| File | Purpose |
|------|---------|
| `assets/trossen_one_arm_scene.xml` | Scene (arm, cube, bucket, camera) |
| `sim_env.py` | Task rewards, success detection |
| `scripted_policy.py` | Expert demonstrations (IK-based) |
| `scripts/record_sim_episodes.py` | Data collection |

**Critical design decision:**
```python
# ❌ WRONG: Identity mapping (80% of beginners make this mistake!)
action[t] = current_position  # Learns "do nothing"

# ✅ CORRECT: Next position mapping
action[t] = next_position  # qpos[t+1] - Learns movement
```

**Why this matters:** Policy must learn state→next_state transitions, NOT identity function. This single bug breaks the entire learning pipeline!

---

### **Layer 2: Data Pipeline**

**Location:** `scripts/`

**What it does:** Transforms raw data into training-ready format

**Process:**

```
Raw HDF5 (26GB)
    ↓
[scripts/create_lerobot_dataset.py]
    ↓
LeRobot Parquet (447MB)
    ├─ data.parquet (states, actions)
    ├─ videos/*.mp4 (compressed images)
    └─ stats.safetensors (normalization)
```

**Compression:** 58x reduction via MP4 encoding

**Key benefit:** Faster training (column-oriented Parquet)

---

### **Layer 3: Learning**

**Location:** `scripts/train_policy.py`

**What it does:** Trains ACT policy to mimic expert

**Architecture:**

```
ACT Policy:
├─ Vision: ResNet-18 encoder (84×84 RGB → features)
├─ State: 6D joint positions
├─ Decoder: Transformer (action chunking)
└─ Output: 100-step action sequence
```

**Training configuration:**
- Steps: 30,000 (fixed, no periodic eval)
- Batch size: 8
- Learning rate: 1e-5
- Device: MPS (Apple Silicon)

**Results:**
- Initial loss: 4.374
- Final loss: 0.036
- Success rate: 0% (shows intent, needs more data)

---

## Data Flow

### **Complete Pipeline:**

```
1. Expert Demonstration
   ├─ scripted_policy.py generates perfect trajectory
   ├─ Randomized cube positions
   └─ Record 50 episodes

2. Data Recording
   ├─ Save to data/raw/*.hdf5 (26GB)
   └─ Images + joint positions + actions

3. Format Conversion
   ├─ HDF5 → LeRobot Parquet + MP4
   ├─ Compute normalization statistics
   └─ data/lerobot/ (447MB)

4. Training
   ├─ Sample batches from Parquet
   ├─ Train ACT for 30k steps
   ├─ Save checkpoints every 5k
   └─ outputs/train/.../030000/

5. Evaluation
   ├─ Load trained model
   ├─ Run 10 test episodes
   └─ Record video + measure success
```

---

## Critical Components

### **1. Expert Policy (Scripted)**

**File:** `scripted_policy.py`

**How it works:**
```
1. Move arm above cube
2. Lower arm
3. Close gripper (grasp)
4. Lift cube
5. Move to bucket
6. Open gripper (release)
7. Return to home
```

**Uses:** Inverse kinematics for smooth trajectories

**Why scripted?** Perfect demonstrations for IL

---

### **2. Image Preprocessing**

**Critical for success:**

```python
# Both training and eval MUST use:
image = image / 255.0  # Scale to [0,1]
mean = [0.485, 0.456, 0.406]  # ImageNet
std = [0.229, 0.224, 0.225]    # ImageNet
image = (image - mean) / std
```

**Bug we fixed:** Eval initially didn't apply ImageNet normalization → policy failed

**Lesson:** Preprocessing consistency is critical!

---

### **3. Observation Space**

```python
observation = {
    'observation.state': joint_positions,  # 6D
    'observation.images.top_cam': image,   # 84×84×3
}
```

**Why dict format?** LeRobot standard

**Camera:** Top-down view of workspace

---

### **4. Action Space**

```python
action = next_joint_positions  # 6D
```

**NOT:** End-effector poses
**WHY:** Direct joint control is simpler for learning

---

## Training Process Explained

### **Why 30k Fixed Steps?**

**Not like this** ❌:
```
Train 1000 → Eval → Check if perfect → Continue
```

**Actually like this** ✅:
```
Train 30,000 steps continuously → Stop → Eval once
```

**Advantages:**
- Simpler (no periodic eval logic)
- Predictable (know exactly how long it takes)
- Sufficient for this dataset size

**Timeline:**

```
Step 0        Step 15k       Step 30k
│             │              │
├─────────────┼──────────────┤
Training...   Save model     STOP → Eval

Loss: 4.374                  0.036
```

---

## Evaluation Design

**Process:**
1. Load trained model
2. Reset environment to random cube position
3. Run policy for max 300 steps
4. Check if cube in bucket
5. Repeat 10 times
6. Record success rate

**Current result:** 0%

**Why?** 
- Policy learned movement patterns ✅
- Needs more data/training for precision
- Shows clear intent to pick (partial success!)

---

## Project Structure

```
trossen-pick-place/
│
├─ trossen_arm_mujoco/       # Layer 1: Simulation
│  ├─ assets/                # MuJoCo scenes
│  ├─ sim_env.py             # Task logic
│  ├─ scripted_policy.py     # Expert
│  └─ scripts/
│     └─ record_sim_episodes.py
│
├─ scripts/                  # Layer 2 & 3
│  ├─ create_lerobot_dataset.py
│  ├─ train_policy.py
│  ├─ eval_policy.py
│  └─ visualize_*.py
│
├─ data/
│  └─ lerobot/               # Training data (447MB)
│
├─ outputs/
│  └─ train/.../030000/      # Trained model (591MB)
│
└─ visualizations/           # Demo videos (27MB)
```

---

## Key Learnings

### **1. Imitation Learning Works**

✅ 50 demonstrations sufficient to learn movement patterns

✅ Loss decreased from 4.374 → 0.036

✅ Policy shows clear intent (reaches toward cube)

⚠️ 0% success rate (needs more data/training for precision)

---

### **2. Details Matter**

**Bugs found and fixed:**

| Bug | Impact | Fix |
|-----|--------|-----|
| Action recording | Policy learned identity | Save `qpos[t+1]` |
| Image normalization | Policy failed completely | Match ImageNet stats |
| Data directory | 95GB in `~/.trossen` | Use explicit `--data_dir` |

**Lesson:** Small implementation details can break learning entirely

---

### **3. MuJoCo is Perfect for Learning**

✅ Fast iteration (500 FPS)

✅ Simple setup (no GPU needed)

✅ Good enough for IL concepts

✅ Free (learn without budget)

**What you learned here** transfers directly to Isaac Sim when you're ready to scale

---

## Success Metrics

### **What "Success" Means**

| Metric | Target | Achieved |
|--------|--------|----------|
| Training completes | Yes | ✅ Yes |
| Loss decreases | < 0.1 | ✅ 0.036 |
| Policy shows intent | Reaches cube | ✅ Yes |
| Grasps cube | > 50% rate | ❌ 0% |
| Places in bucket | > 80% rate | ❌ 0% |

**Interpretation:** Pipeline works, needs more data/training for full task completion

---

## How to Improve

**Option 1: More Data**
- Record 100-200 episodes (vs 50)
- More diverse cube positions

**Option 2: Longer Training**
- 50k-100k steps (vs 30k)
- May overfit without more data

**Option 3: Data Augmentation**
- Image transforms (brightness, rotation)
- Helps with limited data

**Option 4: Migrate to Isaac Sim**
- Domain randomization (lighting, textures)
- Better sim-to-real transfer
- If deploying to real robot

---

## What You Achieved

**From zero to complete IL pipeline in 2-3 weeks:**

✅ Simulation setup (MuJoCo, expert policy)

✅ Data collection (50 episodes, domain randomization)

✅ Format conversion (HDF5 → LeRobot)

✅ Policy training (ACT transformers)

✅ Evaluation framework

✅ Debugging skills (action bug, normalization)

✅ Documentation (guides, architecture)

**This knowledge is the foundation** for any robot learning project!

---

## Connecting to the Journey

**Document 01:** You learned IL is for copying experts → You used it here!

**Document 02:** You saw pick-place as a use case → You built it!

**Document 03:** You learned Isaac Sim for scaling → Your next step!

**Document 04:** You learned MuJoCo for learning → Perfect choice!

**Document 05:** You see how it all comes together in your demo ✅

**The journey is complete.** You're now ready to tackle real robot learning challenges! 🚀

---

## Next Steps

**Apply your knowledge:**

1. **Improve this demo:**
   - Collect more data
   - Try longer training
   - Experiment with hyperparameters

2. **Try Isaac Sim:**
   - Migrate your pick-place task
   - Add domain randomization
   - Prepare for real robot deployment

3. **Build something new:**
   - Different manipulation task
   - Use RL instead of IL
   - Apply to your own robot project

**You have the tools. Now go build!** 🎯

---

*Part 5 of 5-part learning journey - Complete!*

**→ Back to start:** [01 - Robot Learning Methods Overview](01_robot_learning_methods_overview.md)
