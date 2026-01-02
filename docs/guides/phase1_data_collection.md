# Phase 1: Data Collection & Expert Demonstration

## Objective
Generate 50 expert demonstration episodes using a scripted policy and create the expert demo video.

## Prerequisites
```bash
# Install dependencies
pip install lerobot mujoco dm_control h5py opencv-python

# Verify installation
python -c "import mujoco; import lerobot; print('✓ Dependencies installed')"
```

## Steps

### 1. Record Expert Demonstrations

Record 50 episodes of the expert policy successfully picking and placing the cube:

```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
  --task_name sim_pick_place \
  --data_dir data/raw \
  --num_episodes 50 \
  --cam_names cam_high
```

**Expected output:**
- `data/raw/episode_*.hdf5` (50 files, ~26GB total)
- Console log showing 50/50 successful episodes

**⏱️ Time:** ~10-15 minutes

---

### 2. Convert to LeRobot Format

Convert HDF5 files to LeRobot's Parquet format with embedded videos:

```bash
python scripts/create_lerobot_dataset.py \
  --data_dir data/raw \
  --output_dir data/lerobot
```

**Expected output:**
- `data/lerobot/sim_pick_place_demo/` directory
- Parquet files + MP4 videos (~447MB total)

**⏱️ Time:** ~5-10 minutes

---

### 3. Generate Expert Demo Video

Create a visualization of the expert policy in action:

```bash
python scripts/visualize_expert_demo.py
```

**Expected output:**
- `visualizations/expert_demo.mp4` (~2MB)
- Shows successful pick-and-place behavior

**⏱️ Time:** ~1 minute

---

## Verification

Check that data was successfully created:

```bash
# Verify HDF5 files
ls -lh data/raw/*.hdf5 | wc -l  # Should show 50

# Verify LeRobot dataset
ls -lh data/lerobot/sim_pick_place_demo/

# Verify video
open visualizations/expert_demo.mp4  # macOS
```

## Next Steps

✅ Phase 1 complete! Proceed to [Phase 2: Training](phase2_training.md)
