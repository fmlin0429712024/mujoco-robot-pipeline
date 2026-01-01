
import gymnasium as gym
import sys
import os
import cv2
import numpy as np

# Add project root needed for gym env import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv
from gymnasium.envs.registration import register

# Check if already registered
if "trossen_arm/OneArmPickPlace-v0" not in gym.envs.registry:
    print("Registering trossen_arm/OneArmPickPlace-v0")
    register(
        id="trossen_arm/OneArmPickPlace-v0",
        entry_point="trossen_arm_mujoco.gym_env:TrossenGymEnv",
        max_episode_steps=600,
    )

def visualize_random_policy(output_path="before_training.mp4"):
    env = gym.make("trossen_arm/OneArmPickPlace-v0", render_mode="rgb_array")
    
    # Reset
    obs, info = env.reset(seed=42)
    
    frames = []
    
    # Run for 100 steps
    for _ in range(200):
        # Human readable render not supported by wrapper logic directly, 
        # but wrapper render() returns rgb_array if render_mode is rgb_array.
        frame = env.render()
        if frame is not None:
             frames.append(frame)
        
        # Random action
        action = env.action_space.sample()
        # Scale action for smoother movement? Or just random.
        # Random is fine for "Before" state - shows lack of control.
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
            
    env.close()
    
    if not frames:
        print("No frames captured.")
        return

    # Save video
    h, w, c = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    for frame in frames:
        # RGB to BGR
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    out.release()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_random_policy("visualizations/random_policy.mp4")
