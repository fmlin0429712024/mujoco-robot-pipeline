
import argparse
import h5py
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def create_dataset(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    repo_id = args.repo_id

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory {output_dir} already exists. Use --overwrite to overwrite.")
            return

    # Define features
    features = {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper_l", "gripper_r"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["joint_0_pos", "joint_1_pos", "joint_2_pos", "joint_3_pos", "joint_4_pos", "joint_5_pos", "gripper_l_pos", "gripper_r_pos"]
        },
        "observation.images.top_cam": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    }
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        features=features,
        root=output_dir,
        use_videos=True
    )
    
    # We do NOT add timestamp or task here to avoid validation errors if LeRobot schema logic is strict.
    # LeRobotDataset usually auto-manages timestamp/index.

    hdf5_files = sorted(list(data_dir.glob("episode_*.hdf5")))
    print(f"Found {len(hdf5_files)} HDF5 files.")

    for file_path in tqdm(hdf5_files):
        with h5py.File(file_path, "r") as f:
            length = f["action"].shape[0]
            actions = f["action"][:]
            qpos = f["observations/qpos"][:]
            
            has_image = "observations/images/cam_high" in f
            if has_image:
                imgs = f["observations/images/cam_high"][:]
            
            for i in range(length):
                frame = {
                    "action": torch.from_numpy(actions[i]),
                    "observation.state": torch.from_numpy(qpos[i]),
                    "task": "Pick up the red cube and place it in the green bucket."
                }
                
                if has_image:
                    # HDF5 is (T, H, W, C). imgs[i] is (H, W, C).
                    # LeRobot expects (C, H, W) for add_frame
                    img_np = imgs[i]
                    frame["observation.images.top_cam"] = torch.from_numpy(img_np).permute(2, 0, 1)
                
                dataset.add_frame(frame)
            
            dataset.save_episode()

    # dataset.save_dataset_meta()
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--repo_id", default="local/sim_pick_place_demo", help="Repo ID for the dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    
    args = parser.parse_args()
    create_dataset(args)
