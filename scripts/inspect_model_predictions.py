
import sys
import os
import numpy as np
import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add lerobot to path
lerobot_path = "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages"
sys.path.append(lerobot_path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from trossen_arm_mujoco.inference_client import InferenceClient


def inspect_predictions(
    checkpoint_dir: str = None, 
    repo_id: str = "sim_pick_place_demo", 
    root: str = "/Users/folin/projects/trossen-pick-place/local/sim_pick_place_demo",
    num_samples: int = 100,
    inference_mode: str = None,
):
    """
    Inspect model predictions on random dataset samples.
    
    Args:
        checkpoint_dir: Checkpoint path (for local mode)
        repo_id: Dataset repository ID
        root: Dataset root path
        num_samples: Number of samples to inspect
        inference_mode: "triton" or "local" (defaults to env var)
    """
    print(f"Initializing inference client...")
    client = InferenceClient.create(
        mode=inference_mode,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"Loading dataset {repo_id} from {root}...")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    
    # Pick random samples
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    velocities = []
    
    print("Running inference on samples...")
    for idx in tqdm.tqdm(indices):
        item = dataset[idx]
        
        # Convert to numpy for inference client
        obs = {
            "observation.state": item["observation.state"].cpu().numpy(),
            "observation.images.top_cam": item["observation.images.top_cam"].cpu().numpy(),
        }
        
        # Note: Dataset images are already normalized, need to denormalize first
        # For simplicity, we'll use the raw format expected by client
        # The client expects HWC format [H, W, 3] in 0-255 range
        
        # Denormalize ImageNet normalization
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image_normalized = obs["observation.images.top_cam"]  # [3, H, W]
        image_01 = image_normalized * imagenet_std + imagenet_mean
        image_255 = (image_01 * 255.0).astype(np.uint8)
        
        # Convert CHW to HWC
        image_hwc = np.transpose(image_255, (1, 2, 0))
        
        obs["observation.images.top_cam"] = image_hwc
        
        action_np = client.predict(obs)
        velocities.append(action_np)
        
    velocities = np.array(velocities)
    
    client.close()
    
    print("\n--- Prediction Analysis ---")
    print(f"Sample Count: {num_samples}")
    print(f"Action Mean: {velocities.mean(axis=0)}")
    print(f"Action Std: {velocities.std(axis=0)}")
    print(f"Action Max: {velocities.max(axis=0)}")
    print(f"Action Min: {velocities.min(axis=0)}")
    
    # Check if mostly zero
    max_abs = np.max(np.abs(velocities), axis=0)
    print(f"Max Abs Value per Joint: {max_abs}")
    
    zero_joints = np.where(max_abs < 0.01)[0]
    if len(zero_joints) > 0:
        print(f"WARNING: Joints {zero_joints} seem to be outputting near-zero actions.")
    else:
        print("Model seems to be predicting non-zero actions.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model",
        help="Checkpoint directory (for local mode)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["triton", "local"],
        default=None,
        help="Inference mode (defaults to INFERENCE_MODE env var)"
    )
    args = parser.parse_args()
    
    inspect_predictions(
        checkpoint_dir=args.ckpt,
        inference_mode=args.mode,
    )
