# Hidden Directory Data Storage - Explained

## What Happened?

During this project, we discovered **95GB** of data stored in a hidden directory (`~/.trossen`) outside the project folder. This document explains why this happened and how to avoid it.

---

## The Problem

### Initial Setup (Incorrect)

When we first ran the recording script **without** the `--root_dir` argument:

```bash
# ❌ This saved data to ~/.trossen (hidden directory)
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
  --task_name sim_pick_place \
  --num_episodes 50
```

**Result:** Data was saved to `~/.trossen/sim_pick_place/` instead of the project directory.

---

### Fixed Setup (Correct)

After adding `--root_dir .` or `--data_dir data/raw`:

```bash
# ✅ This saves data to project directory
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
  --task_name sim_pick_place \
  --data_dir data/raw \
  --num_episodes 50
```

**Result:** Data correctly saved to `data/raw/` inside the project.

---

## Why Does This Happen?

Many robotics/ML libraries use **default cache directories** in the user's home folder:

| Library | Default Cache | Purpose |
|---------|---------------|---------|
| `trossen_arm_mujoco` | `~/.trossen/` | Episode recordings |
| `lerobot` | `~/.lerobot/` | Dataset cache |
| `huggingface` | `~/.cache/huggingface/` | Model weights |
| `mujoco` | `~/.mujoco/` | MuJoCo assets |

**This is normal behavior** - but can be problematic because:
1. **Hidden directories** (start with `.`) are not visible by default
2. **Large files** accumulate outside the project
3. **Hard to track** disk usage

---

## How to Find Hidden Directories

### Method 1: Check Disk Usage

```bash
# List all hidden directories in home folder with sizes
du -sh ~/.* 2>/dev/null | sort -h
```

### Method 2: Check Specific Known Locations

```bash
# Check robotics-related cache directories
du -sh ~/.trossen ~/.lerobot ~/.mujoco ~/.cache/huggingface 2>/dev/null
```

### Method 3: Find Large Files

```bash
# Find all files larger than 1GB in home directory
find ~ -type f -size +1G 2>/dev/null
```

---

## Lessons Learned

### ✅ Best Practices

1. **Always specify data directories explicitly:**
   ```bash
   --data_dir data/raw
   --output_dir outputs/
   --root_dir .
   ```

2. **Check hidden directories periodically:**
   ```bash
   du -sh ~/.* 2>/dev/null | grep "[0-9]G"  # Show dirs > 1GB
   ```

3. **Add to `.gitignore`:**
   ```
   data/raw/*
   outputs/train/*
   ~/.trossen  # Won't help but documents the issue
   ```

4. **Document in README:**
   - Mention potential hidden directory usage
   - Provide cleanup instructions

---

## Cleanup Checklist

After completing the project:

- [ ] Delete project data: `rm -rf data/raw/*`
- [ ] Delete intermediate checkpoints
- [ ] **Check hidden directories:** `du -sh ~/.trossen ~/.lerobot`
- [ ] **Delete if found:** `rm -rf ~/.trossen ~/.lerobot`
- [ ] Verify disk space: `df -h`

---

## Summary

**What happened:**
- Initial recording saved 95GB to `~/.trossen` (hidden directory)
- This was outside the project and hard to find
- We discovered it when disk space reached 99%

**The fix:**
- Added `--root_dir .` to recording commands
- Explicitly specify `--data_dir data/raw`
- Documented in cleanup guide

**Prevention:**
- Always use explicit path arguments
- Check hidden directories regularly
- Document default cache locations

**Impact:**
- Freed **95GB** of disk space
- Project now saves data correctly in `data/raw/`
- Future users warned in cleanup guide

This is a common pitfall in robotics/ML projects - you're not the first to encounter it! 🎯
