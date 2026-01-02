# Cleanup Guide

## Objective
Free up disk space after completing the learning pipeline by removing large intermediate files.

## What Gets Deleted

After completing all 3 phases, you'll have accumulated ~30GB+ of data:

| Directory | Size | Keep? |
|-----------|------|-------|
| `data/raw/` | ~26GB | ❌ Delete (redundant HDF5 files) |
| `data/lerobot/` | ~447MB | ✅ Keep (needed for future training) |
| `outputs/train/.../checkpoints/005000-025000/` | ~3GB | ❌ Delete (intermediate checkpoints) |
| `outputs/train/.../checkpoints/030000/` | ~591MB | ✅ Keep (final trained model) |
| `visualizations/` | ~27MB | ✅ Keep (your results!) |
| `training.log` | ~13KB | ✅ Keep (training history) |

**Total space to free:** ~29GB

---

## Cleanup Steps

### 1. Delete Raw HDF5 Files

The raw recordings are redundant after converting to LeRobot format:

```bash
rm -rf data/raw/*
```

**Space freed:** ~26GB

---

### 2. Delete Intermediate Checkpoints

Keep only the final 30k checkpoint:

```bash
rm -rf outputs/train/act_pick_place_30k/checkpoints/{005000,010000,015000,020000,025000}
```

**Space freed:** ~3GB

---

### 3. Verify Cleanup

Check that essential files remain:

```bash
# Final checkpoint should still exist
ls -lh outputs/train/act_pick_place_30k/checkpoints/030000/

# LeRobot dataset should still exist
ls -lh data/lerobot/sim_pick_place_demo/

# Videos should still exist
ls -lh visualizations/

# Check total disk usage
du -sh data/ outputs/ visualizations/
```

**Expected remaining size:** ~1.1GB (instead of ~30GB)

---

### 4. Check Disk Space

Verify that space was freed:

```bash
df -h .
```

You should see approximately **29GB more available space** than before cleanup.

---

## Optional: Delete Old Training Runs

If you have multiple training runs:

```bash
# List all training runs
ls -d outputs/train/*/

# Delete specific old run (example)
rm -rf outputs/train/act_pick_place_10k/
```

---

## What to Keep for Future Use

After cleanup, your project should retain:

✅ **Essential files:**
- Trained model: `outputs/train/act_pick_place_30k/checkpoints/030000/`
- Dataset: `data/lerobot/sim_pick_place_demo/`
- Videos: `visualizations/*.mp4`
- Training log: `training.log`
- Source code: All `scripts/` and `trossen_arm_mujoco/`

✅ **Total size:** ~1.1GB (lean and publishable!)

---

## Ready to Share!

Your project is now:
- ✅ Complete (all 3 phases)
- ✅ Clean (minimal disk usage)
- ✅ Ready to push to GitHub
- ✅ Ready for others to reproduce

Push to GitHub:
```bash
git add .
git commit -m "Complete clean pipeline with documentation"
git push origin main
```
