# Documentation Index

> **Quick navigation to all project documentation**

---

## 📖 Core Documentation

**Start here for understanding the project:**

1. **[Architecture Overview](ARCHITECTURE.md)** ⭐
   - Complete pipeline in 5 minutes
   - Training-eval loop explained
   - Component breakdown

2. **[MuJoCo vs Isaac Sim Comparison](mujoco_vs_isaac_sim.md)** ⭐
   - What you learned from MuJoCo
   - Why Isaac Sim for real robots
   - Migration path

3. **[Robot Learning Methods Comparison](robot_learning_methods_comparison.md)** ⭐
   - IL vs RL vs Classical methods
   - SLAM explained
   - Decision framework

---

## 🚀 Strategic Vision

**Big-picture thinking and future plans:**

1. **[Isaac Sim Platform Strategy](isaac_sim_platform_strategy.md)** ⭐⭐⭐
   - 4-phase evolution roadmap (AMR→ARM→LEGS→VLA)
   - ROI: $3.79M from $40K investment
   - Reusable infrastructure across all phases

2. **[Learning Journey Summary](learning_journey_summary.md)** ⭐⭐
   - Complete discussion summary
   - From pick-place IL to AMR navigation
   - Technical and business insights

---

## 📚 Step-by-Step Guides

**Reproduce the project (in `guides/` folder):**

1. **[Phase 1: Data Collection](guides/phase1_data_collection.md)** (~25 min)
   - Record 50 expert episodes
   - Convert to LeRobot format
   - Generate expert demo video

2. **[Phase 2: Training](guides/phase2_training.md)** (~2-3 hours)
   - Generate baseline videos
   - Train ACT policy (30k steps)
   - Monitor training progress

3. **[Phase 3: Deployment](guides/phase3_deployment.md)** (~10 min)
   - Run trained policy in simulation
   - Record deployment video
   - Compare results

4. **[Cleanup Guide](guides/cleanup.md)** (~2 min)
   - Free ~29GB disk space
   - Keep only essential files

---

## 📦 Archive

**Historical/debugging docs (in `archive/` folder):**

- `SYSTEM_DESIGN_DETAILED.md` - Earlier detailed version (superseded by ARCHITECTURE.md)
- `hidden_directory_explained.md` - Debugging note about ~/.trossen issue (resolved)

---

## 🎯 Quick Start

**New to this project?**

1. Read [Architecture Overview](ARCHITECTURE.md) (5 min)
2. Follow [Phase 1 Guide](guides/phase1_data_collection.md) to collect data
3. Check [Isaac Sim Strategy](isaac_sim_platform_strategy.md) for future vision

**Want to understand robotics methods?**

1. Read [Robot Learning Methods Comparison](robot_learning_methods_comparison.md)
2. Read [MuJoCo vs Isaac Sim](mujoco_vs_isaac_sim.md)
3. Read [Learning Journey Summary](learning_journey_summary.md) for full context

---

## 📁 Documentation Structure

```
docs/
├── README.md (this file)
│
├── Core Documentation (root)
│   ├── ARCHITECTURE.md ⭐
│   ├── mujoco_vs_isaac_sim.md ⭐
│   └── robot_learning_methods_comparison.md ⭐
│
├── Strategic Vision (root)
│   ├── isaac_sim_platform_strategy.md ⭐⭐⭐
│   └── learning_journey_summary.md ⭐⭐
│
├── guides/ (step-by-step)
│   ├── phase1_data_collection.md
│   ├── phase2_training.md
│   ├── phase3_deployment.md
│   └── cleanup.md
│
└── archive/ (historical)
    ├── SYSTEM_DESIGN_DETAILED.md
    └── hidden_directory_explained.md
```

---

## ⭐ Priority Reading

**If you only have time for 3 documents:**

1. ✅ **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand the project
2. ✅ **[isaac_sim_platform_strategy.md](isaac_sim_platform_strategy.md)** - Understand the vision
3. ✅ **[robot_learning_methods_comparison.md](robot_learning_methods_comparison.md)** - Understand the methods

**Total reading time:** ~15-20 minutes

---

*Document Structure Last Updated: January 2, 2026*
