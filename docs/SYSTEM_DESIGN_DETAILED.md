# System Design Document: Robot Learning Pipeline

> **Purpose:** Complete architecture and design reference for the Trossen Arm pick-and-place learning system

---

## 1. System Overview

### 1.1 What This System Does

This system implements an **end-to-end robot learning pipeline** that:
1. Collects expert demonstration data in simulation
2. Converts data to a standardized format
3. Trains a vision-based policy using imitation learning
4. Evaluates the learned policy in simulation

**Task:** Pick up a red cube and place it in a green bucket using a 6-DOF robot arm.

---

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ROBOT LEARNING PIPELINE                     │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   SIMULATION     │      │  DATA PIPELINE   │      │  POLICY LEARNING │
│   ENVIRONMENT    │──────▶│                  │──────▶│                  │
│                  │      │                  │      │                  │
│ • MuJoCo scene   │      │ • HDF5 → Parquet │      │ • ACT training   │
│ • Robot model    │      │ • Video encoding │      │ • LeRobot        │
│ • Task logic     │      │ • Normalization  │      │ • Checkpointing  │
│ • Expert policy  │      │                  │      │                  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         │                         │                         │
         ▼                         ▼                         ▼
   data/raw/*.hdf5      data/lerobot/dataset/      outputs/train/
                                                     checkpoints/
```

---

## 2. System Components

### 2.1 Component Hierarchy

```
trossen-pick-place/
│
├── LAYER 1: SIMULATION CORE
│   └── trossen_arm_mujoco/
│       ├── assets/              # MuJoCo scene definitions
│       ├── sim_env.py           # Task environments
│       ├── scripted_policy.py   # Expert demonstrations
│       └── scripts/
│           └── record_sim_episodes.py  # Data collection
│
├── LAYER 2: DATA PIPELINE
│   └── scripts/
│       ├── create_lerobot_dataset.py   # Format conversion
│       └── visualize_*.py              # Visualization tools
│
├── LAYER 3: LEARNING
│   └── scripts/
│       ├── train_policy.py     # Policy training
│       └── eval_policy.py      # Policy evaluation
│
└── LAYER 4: DATA STORAGE
    ├── data/
    │   ├── raw/                # Raw recordings (HDF5)
    │   └── lerobot/            # Processed dataset (Parquet)
    └── outputs/
        └── train/              # Trained models
```

---

## 3. Detailed Component Breakdown

### 3.1 Layer 1: Simulation Core

**Location:** `trossen_arm_mujoco/`

#### **Purpose:**
Provides the physics simulation, robot model, and task definition.

#### **Key Files:**

| File | Purpose | What It Does |
|------|---------|--------------|
| `assets/trossen_one_arm_scene.xml` | Scene definition | Defines robot, bucket, cube, camera, lighting |
| `sim_env.py` | Task environment | Implements pick-place reward, success detection |
| `scripted_policy.py` | Expert policy | Generates optimal trajectories via inverse kinematics |
| `utils.py` | Environment factory | Creates simulation instances |
| `gym_env.py` | Gym wrapper | Standardizes interface for LeRobot |
| `scripts/record_sim_episodes.py` | Data collection | Executes expert policy, saves observations + actions |

#### **Data Flow:**
```
MuJoCo Scene → Expert Policy → Observations + Actions → HDF5 Files
```

#### **Critical Design Decisions:**

1. **Action Space:** Joint positions (6-DOF), not end-effector poses
   - Why: More direct control, easier for learning
   - Where: `sim_env.py` defines `action_space`

2. **Action Recording Fix:**
   ```python
   # CRITICAL FIX in record_sim_episodes.py
   action[t] = qpos[t+1]  # Target position (future state)
   # NOT: action[t] = qpos[t]  # Current position (identity map)
   ```
   - Why: Policy needs to learn state → next_state mapping
   - Impact: Without this fix, policy learns to output current state (identity)

3. **Observation Space:**
   - State: 6D joint positions
   - Visual: 84×84 RGB image from top camera
   - Where: `gym_env.py` defines observation dict

---

### 3.2 Layer 2: Data Pipeline

**Location:** `scripts/`

#### **Purpose:**
Transform raw simulation data into training-ready format.

#### **Key Files:**

| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `create_lerobot_dataset.py` | `data/raw/*.hdf5` | `data/lerobot/` | Convert to LeRobot Parquet + MP4 format |
| `visualize_expert_demo.py` | Simulation | `expert_demo.mp4` | Generate expert video |
| `visualize_random_policy.py` | Simulation | `random_policy.mp4` | Generate baseline video |
| `visualize_untrained_policy.py` | Model | `untrained_policy.mp4` | Show pre-training behavior |

#### **Data Transformation Pipeline:**

```
HDF5 Episode Files (26GB)
    ↓
[create_lerobot_dataset.py]
    ↓
LeRobot Dataset (447MB)
    ├── data.parquet         # States, actions (tabular)
    ├── videos/              # Compressed MP4s
    ├── stats.safetensors    # Normalization statistics
    └── meta/                # Episode metadata
```

#### **Critical Design Decisions:**

1. **Why Convert HDF5 → Parquet?**
   - HDF5: Good for recording (sequential writes)
   - Parquet: Better for training (column-oriented, faster random access)
   - Video compression: 26GB → 447MB (58x size reduction)

2. **Normalization Statistics:**
   - Computed during conversion
   - Saved in `stats.safetensors`
   - Used during training and evaluation
   - **Gap identified:** Must ensure eval uses same stats as training

3. **Video Encoding:**
   - Format: MP4 (H.264)
   - Backend: PyAV
   - Why: Efficient storage, LeRobot compatible

---

### 3.3 Layer 3: Learning

**Location:** `scripts/`

#### **Purpose:**
Train and evaluate vision-based policies using imitation learning.

#### **Key Files:**

| File | Purpose | Framework |
|------|---------|-----------|
| `train_policy.py` | Train ACT policy | LeRobot + PyTorch |
| `eval_policy.py` | Evaluate trained policy | Custom evaluation loop |

#### **Training Pipeline:**

```
LeRobot Dataset
    ↓
[train_policy.py]
    │
    ├─▶ ACT Policy Architecture
    │   ├── ResNet-18 (vision encoder)
    │   ├── Transformer (action sequence decoder)
    │   └── VAE (latent action encoding)
    │
    ├─▶ Training Loop
    │   ├── Sample batches
    │   ├── Compute loss
    │   ├── Backprop + optimize
    │   └── Save checkpoints every 5k steps
    │
    └─▶ Output
        └── outputs/train/act_pick_place_30k/
            └── checkpoints/
                └── 030000/
                    └── pretrained_model/
```

#### **Critical Design Decisions:**

1. **Policy Architecture: ACT (Action Chunking with Transformers)**
   - Input: Image (84×84×3) + State (6D)
   - Output: Action sequence (100 steps ahead)
   - Why: Temporal consistency, smoother trajectories

2. **Training Configuration:**
   ```python
   steps = 30000
   batch_size = 8
   learning_rate = 1e-5
   device = "mps"  # Apple Silicon
   ```
   - Why 30k steps: Empirically sufficient for 50 episodes
   - Why batch_size=8: Memory constraint on M-series chip

3. **Image Preprocessing (CRITICAL):**
   ```python
   # Training (in LeRobot):
   image = normalize_imagenet(image)  # mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
   
   # Evaluation (in eval_policy.py):
   image = (image / 255.0 - imagenet_mean) / imagenet_std  # MUST MATCH
   ```
   - **Gap identified:** Early evaluations failed because eval didn't apply ImageNet normalization
   - **Fix applied:** Added explicit ImageNet normalization in `eval_policy.py`

4. **Checkpoint Strategy:**
   - Save every 5k steps
   - Keep all for analysis
   - **Post-training cleanup:** Delete intermediates (5k, 10k, 15k, 20k, 25k), keep final 30k

---

### 3.4 Layer 4: Data Storage

**Location:** `data/` and `outputs/`

#### **Structure:**

```
data/
├── raw/                    # Ephemeral (deleted after conversion)
│   └── episode_*.hdf5      # 26GB, not version controlled
│
└── lerobot/
    └── sim_pick_place_demo/
        ├── data.parquet    # Essential for training
        ├── videos/         # Essential for training
        └── stats.safetensors  # Essential for eval

outputs/
└── train/
    └── act_pick_place_30k/
        ├── checkpoints/
        │   └── 030000/     # Final model (591MB)
        └── training.log    # Training history
```

#### **Storage Strategy:**

| Data Type | Size | Lifecycle | Git Tracking |
|-----------|------|-----------|--------------|
| Raw HDF5 | 26GB | Temporary | ❌ .gitignore |
| LeRobot Dataset | 447MB | Permanent | ❌ .gitignore (too large) |
| Trained Model | 591MB | Permanent | ❌ .gitignore (distribute via releases) |
| Source Code | ~10MB | Permanent | ✅ Tracked |
| Documentation | <1MB | Permanent | ✅ Tracked |
| Videos | 27MB | Permanent | ❌ .gitignore (link YouTube instead) |

---

## 4. Data Flow Analysis

### 4.1 End-to-End Flow

```
┌─────────────┐
│ START       │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Phase 1: Data Collection                │
│ └─ record_sim_episodes.py               │
│                                          │
│ MuJoCo → Expert Policy → Observations   │
│ Input:  None                             │
│ Output: data/raw/*.hdf5 (26GB)          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Phase 2a: Data Conversion               │
│ └─ create_lerobot_dataset.py            │
│                                          │
│ HDF5 → Parquet + MP4                    │
│ Input:  data/raw/*.hdf5                 │
│ Output: data/lerobot/ (447MB)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Phase 2b: Policy Training               │
│ └─ train_policy.py                      │
│                                          │
│ Dataset → ACT Model → Checkpoints       │
│ Input:  data/lerobot/                   │
│ Output: outputs/train/.../030000/       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Phase 3: Deployment & Evaluation        │
│ └─ eval_policy.py                       │
│                                          │
│ Model → Simulation → Success Rate       │
│ Input:  outputs/train/.../030000/       │
│ Output: visualizations/after_training.mp4│
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────┐
│ END         │
└─────────────┘
```

---

## 5. Critical System Dependencies

### 5.1 External Libraries

| Library | Version | Purpose | Used In |
|---------|---------|---------|---------|
| `mujoco` | Latest | Physics simulation | `sim_env.py`, `record_sim_episodes.py` |
| `lerobot` | Latest | Dataset format, ACT policy | `create_lerobot_dataset.py`, `train_policy.py` |
| `torch` | 2+ | Deep learning | `train_policy.py`, `eval_policy.py` |
| `h5py` | Latest | HDF5 I/O | `record_sim_episodes.py` |
| `opencv-python` | Latest | Video encoding | `eval_policy.py` |
| `gymnasium` | Latest | RL environment interface | `gym_env.py` |

### 5.2 Internal Dependencies

```
gym_env.py
    ↓ depends on
sim_env.py
    ↓ depends on
assets/*.xml

create_lerobot_dataset.py
    ↓ reads
data/raw/*.hdf5 (from record_sim_episodes.py)

train_policy.py
    ↓ reads
data/lerobot/ (from create_lerobot_dataset.py)

eval_policy.py
    ↓ reads
outputs/train/.../pretrained_model/ (from train_policy.py)
```

---

## 6. Identified Gaps & Improvements

### 6.1 Current Gaps

#### **Gap 1: Hidden Directory Management**
- **Issue:** Recording script can save to `~/.trossen/` if `--root_dir` not specified
- **Impact:** 95GB of orphaned data outside project
- **Fix Applied:** Document in cleanup guide, use explicit `--data_dir`
- **Improvement:** Add validation in `record_sim_episodes.py` to warn if saving to home directory

#### **Gap 2: Image Normalization Consistency**
- **Issue:** Training uses ImageNet stats, early eval didn't
- **Impact:** Policy saw completely different image distributions
- **Fix Applied:** Added ImageNet normalization to `eval_policy.py`
- **Improvement:** Create shared preprocessing module used by both train and eval

#### **Gap 3: Action Recording Bug**
- **Issue:** Original code saved `qpos[t]` as action (identity mapping)
- **Impact:** Policy learns to output current state, not next state
- **Fix Applied:** Changed to `qpos[t+1]`
- **Improvement:** Add unit test to verify action[t] ≠ state[t]

#### **Gap 4: Disk Space Management**
- **Issue:** Users may not realize raw HDF5 can be deleted after conversion
- **Impact:** Waste 26GB of disk space
- **Fix Applied:** Cleanup guide documents this
- **Improvement:** Add automatic cleanup option in conversion script

#### **Gap 5: Evaluation Metrics**
- **Issue:** Only binary success/failure reported
- **Impact:** Hard to diagnose partial progress (e.g., "reaches cube but doesn't grasp")
- **Current State:** 0% success but arm shows intent
- **Improvement:** Add granular metrics:
  - Distance to cube
  - Grasp success rate
  - Lift success rate
  - Placement accuracy

### 6.2 Suggested Architecture Improvements

#### **Improvement 1: Shared Config File**
**Current:** Hardcoded paths in multiple scripts
**Proposed:**
```python
# config.py
DATA_RAW_DIR = "data/raw"
DATA_LEROBOT_DIR = "data/lerobot"
CHECKPOINT_DIR = "outputs/train"
DEVICE = "mps"
```

#### **Improvement 2: Preprocessing Module**
**Current:** Image preprocessing duplicated in train and eval
**Proposed:**
```python
# preprocessing.py
def preprocess_image_for_policy(image):
    """Shared preprocessing for training and evaluation"""
    image = image / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image
```

#### **Improvement 3: Modular Data Collection**
**Current:** `record_sim_episodes.py` does everything
**Proposed:** Separate concerns
```
data_collection/
├── collector.py      # Core recording logic
├── expert_policy.py  # Policy interface
└── storage.py        # HDF5 writing
```

#### **Improvement 4: Experiment Tracking**
**Current:** Manual log inspection
**Proposed:** Add WandB or TensorBoard integration
- Loss curves
- Success rate tracking
- Hyperparameter logging

---

## 7. Best Practices Followed

### 7.1 Software Engineering

✅ **Separation of Concerns:**
- Simulation (Layer 1)
- Data Pipeline (Layer 2)
- Learning (Layer 3)
- Storage (Layer 4)

✅ **Reproducibility:**
- Fixed random seeds
- Documented dependencies
- Version-controlled code

✅ **Documentation:**
- Phase-by-phase guides
- Architecture documentation
- Cleanup instructions

### 7.2 Machine Learning

✅ **Data Management:**
- Raw data → Processed data separation
- Normalization statistics saved
- Train/eval consistency

✅ **Training:**
- Checkpointing every 5k steps
- Loss logging
- Deterministic evaluation seeds

✅ **Evaluation:**
- Fixed seed evaluation
- Video recording for debugging
- Baseline comparisons (random, untrained)

---

## 8. System Limitations

### 8.1 Known Limitations

1. **Sim-Only:** No real robot deployment (sim-to-real gap not addressed)
2. **Limited Data:** Only 50 episodes, single scene configuration
3. **No Domain Randomization:** Fixed lighting, textures, dynamics
4. **Performance:** 0% success rate (shows intent but lacks precision)
5. **Scalability:** Designed for single task, not multi-task learning

### 8.2 Future Extensions

**To Improve Success Rate:**
1. Record more data (100-200 episodes)
2. Add domain randomization
3. Longer training (50k-100k steps)
4. Data augmentation (image transforms)

**To Enable Real Deployment:**
1. Migrate to Isaac Sim (photorealistic)
2. Add domain randomization
3. Real robot interface (ROS 2)
4. Camera calibration pipeline

---

## 9. Usage Patterns

### 9.1 Typical Development Workflow

```bash
# 1. Collect data
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
  --task_name sim_pick_place \
  --data_dir data/raw \
  --num_episodes 50

# 2. Convert data
python scripts/create_lerobot_dataset.py \
  --data_dir data/raw \
  --output_dir data/lerobot

# 3. Train policy
python scripts/train_policy.py  # ~2-3 hours

# 4. Evaluate
python scripts/eval_policy.py \
  --ckpt outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model

# 5. Cleanup
rm -rf data/raw/*
rm -rf outputs/train/.../checkpoints/{005000,010000,015000,020000,025000}
```

### 9.2 Debugging Workflow

**If training loss not decreasing:**
1. Check normalization: `scripts/debug_normalization_stats.py`
2. Inspect dataset: `scripts/inspect_dataset_playback.py`
3. Verify actions: `scripts/verify_joint_order.py`

**If evaluation fails:**
1. Check image preprocessing (ImageNet normalization)
2. Verify action unnormalization
3. Compare with expert demo video

---

## 10. Conclusion

This system implements a complete robot learning pipeline with clear separation of concerns, reproducible workflows, and comprehensive documentation. 

**Strengths:**
- ✅ Modular architecture
- ✅ Well-documented
- ✅ Reproducible
- ✅ Educational value

**Areas for Improvement:**
- 🔧 Consolidate preprocessing logic
- 🔧 Add granular evaluation metrics
- 🔧 Improve success rate (more data/longer training)
- 🔧 Add experiment tracking

**Next Steps:**
- Implement suggested improvements (Section 6.2)
- Increase dataset size (100+ episodes)
- Explore hyperparameter tuning
- Plan Isaac Sim migration for real deployment
