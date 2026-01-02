# 🤖 Trossen Arm Pick & Place with ACT Policy

> **My First Robot Project** - Learning robotics by building an end-to-end imitation learning pipeline

This project demonstrates a complete robotics learning pipeline using the **ACT (Action Chunking with Transformers)** policy from LeRobot to teach a simulated Trossen robot arm to pick up a cube and place it in a bucket.

## 📺 Video Showcase

| Phase | Video | Description |
|-------|-------|-------------|
| **Phase 1** | [Expert Demo](https://youtu.be/VuP907sxELQ) | Scripted expert policy successfully picking and placing |
| **Phase 1** | [Random Policy](https://youtu.be/IaS8G5BYmAQ) | Untrained arm moving randomly (baseline) |
| **Phase 1** | [Before Training](https://youtu.be/tw9J1FFLFPs) | Arm at rest / minimal movement |
| **Phase 2** | [After Training (30k steps)](https://youtu.be/ULep7-XoTZM) | ACT policy attempting pick-and-place |

> 📝 *Videos hosted on YouTube*

---

## 🎯 Project Overview

### The 3-Phase Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PHASE 1       │     │   PHASE 2       │     │   PHASE 3       │
│  Data Collection│────▶│    Training     │────▶│   Deployment    │
│                 │     │                 │     │                 │
│ • MuJoCo sim    │     │ • ACT policy    │     │ • Sim deploy    │
│ • 50 episodes   │     │ • 30k steps     │     │ • Evaluation    │
│ • Expert demos  │     │ • LeRobot       │     │ • Coming soon   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

> 💡 **Next Step:** Interested in Isaac Sim for real robot deployment? See [MuJoCo vs Isaac Sim Comparison](docs/mujoco_vs_isaac_sim.md)

---

## 📁 Project Structure

```
trossen-pick-place/
├── trossen_arm_mujoco/          # MuJoCo simulation environment
│   ├── assets/                  # Robot MJCF models
│   ├── scripts/
│   │   └── record_sim_episodes.py  # Phase 1: Data collection
│   ├── sim_env.py               # Joint-space environment
│   ├── ee_sim_env.py            # End-effector environment
│   ├── scripted_policy.py       # Expert pick-and-place policy
│   └── gym_env.py               # Gymnasium wrapper for LeRobot
├── scripts/
│   ├── create_lerobot_dataset.py   # Convert HDF5 → LeRobot format
│   ├── train_policy.py             # Phase 2: ACT training
│   ├── eval_policy.py              # Evaluate trained policy
│   ├── visualize_expert_demo.py    # Generate expert video
│   ├── visualize_random_policy.py  # Generate random baseline video
│   └── visualize_untrained_policy.py  # Generate untrained video
├── data/
│   ├── raw/                     # Raw HDF5 episode recordings
│   └── lerobot/                 # LeRobot dataset format
├── outputs/
│   └── train/act_pick_place_30k/  # Trained model checkpoints
├── visualizations/              # Generated demo videos
│   ├── expert_demo.mp4
│   ├── random_policy.mp4
│   ├── untrained_policy.mp4
│   └── after_training.mp4
└── README.md
```

---

## 🚀 Quick Start

### Step-by-Step Guides

Follow these guides to reproduce the complete learning pipeline:

1. **[Phase 1: Data Collection](docs/phase1_data_collection.md)** (~25 min)
   - Record 50 expert demonstrations
   - Convert to LeRobot format
   - Generate expert demo video

2. **[Phase 2: Training](docs/phase2_training.md)** (~2-3 hours)
   - Generate baseline videos (random & untrained)
   - Train ACT policy for 30k steps
   - Monitor training progress

3. **[Phase 3: Deployment](docs/phase3_deployment.md)** (~10 min)
   - Run trained policy in simulation
   - Record deployment video
   - Compare results

4. **[Cleanup](docs/cleanup.md)** (~2 min)
   - Free ~29GB disk space
   - Keep only essential files

### Prerequisites

```bash
# Python 3.10+
pip install lerobot mujoco dm_control h5py opencv-python
```

### One-Line Quick Test

```bash
# See the expert policy in action (no training required)
python scripts/visualize_expert_demo.py && open visualizations/expert_demo.mp4
```
*Coming soon...*

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Training Steps | 30,000 |
| Episodes | 50 |
| Batch Size | 8 |
| Final Loss | 0.036 |
| Device | Apple M-series (MPS) |

### Loss Progression

```
Step     Loss     Notes
─────────────────────────────
0k       4.374    Initial
5k       0.073    Rapid drop
10k      0.057    Checkpoint 1
20k      0.043    Continued improvement
30k      0.036    Final model
```

---

## 🎓 What I Learned

### Phase 1: Data Collection
- Recording expert demonstrations in simulation
- Importance of action representation (action[t] = target position, not current)
- HDF5 data storage for robot trajectories

### Phase 2: Training
- ACT (Action Chunking with Transformers) architecture
- LeRobot dataset format and video encoding
- Training on Apple Silicon (MPS device)
- ImageNet normalization for vision models

### Phase 3: Deployment (Upcoming)
- Sim-to-real transfer challenges
- Real-world latency and noise handling
- Safety constraints for physical robots

---

## 🛠️ Key Files Explained

| File | Purpose |
|------|---------|
| `scripted_policy.py` | Expert policy using inverse kinematics |
| `record_sim_episodes.py` | Records expert demos to HDF5 |
| `create_lerobot_dataset.py` | Converts HDF5 → LeRobot Parquet + video |
| `train_policy.py` | Launches ACT training with LeRobot |
| `eval_policy.py` | Evaluates trained policy with video output |
| `gym_env.py` | Gymnasium wrapper for LeRobot compatibility |

---

## 📚 References

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face robotics library
- [ACT Policy](https://arxiv.org/abs/2304.13705) - Action Chunking with Transformers
- [Trossen Robotics](https://www.trossenrobotics.com/) - Robot arm hardware

---

## 📝 License

MIT License - Feel free to use this for learning!

---

*This is my first robot project. Feedback and suggestions welcome! 🤖*
