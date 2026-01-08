
import gymnasium as gym
import numpy as np
import cv2
import sys
import os
import tqdm
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv
from trossen_arm_mujoco.inference_client import InferenceClient


def eval_policy(
    checkpoint_dir: str = "outputs/train/act_pick_place_10k/checkpoints/010000/pretrained_model",
    num_episodes: int = 10,
    output_video: str = "visualizations/after_training.mp4",
    inference_mode: str = None,
    max_steps: int = 600,
):
    """
    Evaluate trained policy using inference client.
    
    Args:
        checkpoint_dir: Path to checkpoint (used in local mode)
        num_episodes: Number of episodes to evaluate
        output_video: Path to save output video
        inference_mode: "triton" or "local" (defaults to env var)
    """
    print(f"Initializing inference client...")
    
    # Create inference client (auto-detects mode from env or uses provided)
    client = InferenceClient.create(
        mode=inference_mode,
        checkpoint_dir=checkpoint_dir,
    )
    
    env = TrossenGymEnv(render_mode="rgb_array")
    
    success_count = 0
    total_rewards = []
    frames = []

    for ep in range(num_episodes):
        print(f"Running Episode {ep+1}/{num_episodes} (Seed {ep})...")
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0
        ep_success = False
        
        # max_steps set by argument
        step = 0
        
        while not done and step < max_steps:
            # Run inference using client (handles all preprocessing)
            action_np = client.predict(obs)
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            # Check success condition
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
    client.close()
    
    success_rate = success_count / num_episodes
    print(f"\nEvaluation Complete.")
    print(f"Success Rate: {success_rate * 100:.2f}% ({success_count}/{num_episodes})")
    
    # Save video
    if frames:
        print(f"Saving video to {output_video}...")
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
            
        video.release()
        print("Video saved.")


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
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--output_video", type=str, default="visualizations/after_training.mp4", help="Output video path")
    
    args = parser.parse_args()
    
    eval_policy(
        checkpoint_dir=args.ckpt,
        num_episodes=args.episodes,
        inference_mode=args.mode,
        max_steps=args.max_steps,
        output_video=args.output_video
    )
