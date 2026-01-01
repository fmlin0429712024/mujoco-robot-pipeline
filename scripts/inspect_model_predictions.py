
import sys
import os
import torch
import numpy as np
import tqdm
from pathlib import Path

# Add lerobot to path
lerobot_path = "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages"
sys.path.append(lerobot_path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

def inspect_predictions(
    checkpoint_dir, 
    repo_id="sim_pick_place_demo", 
    root="/Users/folin/projects/trossen-pick-place/local/sim_pick_place_demo",
    num_samples=100
):
    print(f"Loading policy from {checkpoint_dir}...")
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.eval()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy.to(device)
    print(f"Using device: {device}")

    print(f"Loading dataset {repo_id} from {root}...")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    
    # Pick random samples
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    velocities = []
    
    print("Running inference on samples...")
    for idx in tqdm.tqdm(indices):
        item = dataset[idx]
        
        # Prepare batch [B, ...]
        state = item["observation.state"].unsqueeze(0).to(device)
        image = item["observation.images.top_cam"].unsqueeze(0).to(device)
        
        batch = {
            "observation.state": state,
            "observation.images.top_cam": image
        }
        
        with torch.no_grad():
            action = policy.select_action(batch)
            
        # Action is [B, D] -> [D]
        action_np = action.squeeze(0).cpu().numpy()
        velocities.append(action_np)
        
    velocities = np.array(velocities)
    
    print("\n--- Prediction Analysis ---")
    print(f"Sample Count: {num_samples}")
    print(f"Action Mean: {velocities.mean(axis=0)}")
    print(f"Action Std: {velocities.std(axis=0)}")
    print(f"Action Max: {velocities.max(axis=0)}")
    print(f"Action Min: {velocities.min(axis=0)}")
    
    # Check if mostly zero (heuristic: max abs value < 0.01)
    max_abs = np.max(np.abs(velocities), axis=0)
    print(f"Max Abs Value per Joint: {max_abs}")
    
    zero_joints = np.where(max_abs < 0.01)[0]
    if len(zero_joints) > 0:
        print(f"WARNING: Joints {zero_joints} seem to be outputting near-zero actions.")
    else:
        print("Model seems to be predicting non-zero actions.")

if __name__ == "__main__":
    inspect_predictions(
        checkpoint_dir="outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model"
    )
