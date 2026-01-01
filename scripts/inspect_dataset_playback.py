
import sys
import os
import cv2
import torch
import numpy as np
import tqdm
from pathlib import Path

# Add lerobot to path
lerobot_path = "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages"
sys.path.append(lerobot_path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def inspect_dataset(repo_id, root, output_video, num_episodes=5):
    print(f"Loading dataset {repo_id} from {root}...")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Features: {dataset.features}")
    
    frames = []
    
    # Select random episodes
    indices = np.random.choice(range(dataset.num_episodes), size=min(num_episodes, dataset.num_episodes), replace=False)
    indices.sort()
    
    print(f"Inspecting episodes: {indices}")
    
    for ep_idx in indices:
        # Get frame indices for this episode
        try:
            if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
                 # episodes is a list of {'length': N, ...} usually in LeRobot
                 # We need to manually compute cumulative index
                 # This is O(N) but N=50 is small.
                 start_idx = 0
                 for i in range(ep_idx):
                     start_idx += dataset.meta.episodes[i]["length"]
                 length = dataset.meta.episodes[ep_idx]["length"]
                 end_idx = start_idx + length
            else:
                 # Last resort: random sample of frames if structure unknown
                 print(f"Unknown episode structure. Sampling 100 frames from index 0.")
                 start_idx = 0
                 end_idx = 100
        except Exception as e:
            print(f"Error accessing episode index: {e}")
            continue
        
        print(f"Episode {ep_idx}: Frames {start_idx} to {end_idx} ({end_idx - start_idx} frames)")
        
        for i in range(start_idx, end_idx):
            item = dataset[i]
            
            # Image is C, H, W torch tensor
            img_tensor = item["observation.images.top_cam"]
            # Convert to H, W, C numpy
            img_np = img_tensor.permute(1, 2, 0).numpy()
            # If float [0,1] or int [0,255]? PyAV usually gives uint8 [0,255]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
                
            # Draw Action / State info on image
            # State: 8 dim
            state = item["observation.state"]
            # Action: 8 dim
            action = item["action"]
            
            # Add text
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.putText(img_bgr, f"Ep {ep_idx} Fr {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_bgr, f"State: {state[:4].numpy().round(2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img_bgr, f"Action: {action[:4].numpy().round(2)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            frames.append(img_bgr)
            
    # Save Video
    if frames:
        print(f"Saving video to {output_video}...")
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
        
        for frame in frames:
            video.write(frame)
            
        video.release()
        print("Video saved.")

if __name__ == "__main__":
    inspect_dataset(
        repo_id="sim_pick_place_demo",
        root="/Users/folin/projects/trossen-pick-place/local/sim_pick_place_demo",
        output_video="visualizations/dataset_inspection.mp4"
    )
