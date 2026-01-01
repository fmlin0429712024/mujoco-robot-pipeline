
import gymnasium as gym
import torch
import numpy as np
import cv2
import sys
import os
import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file

def eval_policy(checkpoint_dir="outputs/train/act_pick_place_10k/checkpoints/010000/pretrained_model", num_episodes=10, output_video="visualizations/after_training.mp4"):
    print(f"Loading policy from {checkpoint_dir}...")
    # Load policy
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.eval()
    
    # Check device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    policy.to(device)

    # Load separate stats file for manual unnormalization
    stats_path = os.path.join(checkpoint_dir, "policy_preprocessor_step_3_normalizer_processor.safetensors")
    action_mean = None
    action_std = None
    
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        stats = load_file(stats_path)
        action_mean = stats["action.mean"].to(device)
        action_std = stats["action.std"].to(device)
    else:
        print("WARNING: Stats file not found. Assuming unnormalized output.")

    # Add features to env if missing (mimic training monkeypatch if needed, 
    # but TrossenGymEnv defines spaces which should be enough for basic use)
    # However, policy expects specific keys in forward()
    
    env = TrossenGymEnv(render_mode="rgb_array")
    
    success_count = 0
    total_rewards = []
    
    # Video writer setup
    # We'll save the first episode or all? Let's save all contiguous for now or just first few.
    # To make a nice video, let's stitch all episodes.
    frames = []

    for ep in range(num_episodes):
        print(f"Running Episode {ep+1}/{num_episodes} (Seed {ep})...")
        obs, _ = env.reset(seed=ep) # Use training seeds (0-9) to check memorization
        done = False
        ep_reward = 0
        ep_success = False
        
        # Max steps
        max_steps = 600
        step = 0
        
        while not done and step < max_steps:
            # Prepare observation for policy
            # Policy expects batch dim
            state = torch.from_numpy(obs["observation.state"].copy()).float().unsqueeze(0).to(device)
            image_raw = torch.from_numpy(obs["observation.images.top_cam"].copy()).float()
            # Scale to [0,1] first
            image = image_raw / 255.0
            # Apply ImageNet normalization (must match training)
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - imagenet_mean) / imagenet_std
            image = image.unsqueeze(0).to(device)
            # Image is [B, C, H, W] with ImageNet normalization
            
            batch = {
                "observation.state": state,
                "observation.images.top_cam": image
            }
            
            with torch.no_grad():
                action = policy.select_action(batch)
            
            # Manual Unnormalization
            if action_mean is not None:
                action = action * action_std + action_mean
            
            # Action is [B, D] -> [D]
            action_np = action.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            # Check success condition (reward == 4 means success as per SimEnv)
            if reward == 4:
                ep_success = True
            
            ep_reward += reward
            step += 1
            
            # Capture frame
            frame = env.render()
            frames.append(frame)
            
            if terminated or truncated:
                done = True
        
        if ep_success:
            print(f"Episode {ep+1}: SUCCESS")
            success_count += 1
        else:
            print(f"Episode {ep+1}: FAILURE")
        
        total_rewards.append(ep_reward)

    env.close()
    
    success_rate = success_count / num_episodes
    print(f"\nEvaluation Complete.")
    print(f"Success Rate: {success_rate * 100:.2f}% ({success_count}/{num_episodes})")
    
    # Save video
    if frames:
        print(f"Saving video to {output_video}...")
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        height, width, layers = frames[0].shape
        # mp4v or h264
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
        
        for frame in frames:
            # OpenCV expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
            
        video.release()
        print("Video saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model")
    args = parser.parse_args()
    
    eval_policy(args.ckpt)
