"""
Visualize "just started training" - arm barely moves (near-zero actions).
"""
import gymnasium as gym
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv
from gymnasium.envs.registration import register

if "trossen_arm/OneArmPickPlace-v0" not in gym.envs.registry:
    register(
        id="trossen_arm/OneArmPickPlace-v0",
        entry_point="trossen_arm_mujoco.gym_env:TrossenGymEnv",
        max_episode_steps=600,
    )

def visualize_untrained(output_path="visualizations/untrained_policy.mp4"):
    env = gym.make("trossen_arm/OneArmPickPlace-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    
    frames = []
    
    # Get initial qpos from observation
    initial_state = obs["observation.state"]
    
    for _ in range(200):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Near-zero actions - arm barely moves (simulates untrained policy)
        # Add tiny random noise to show it's "trying" but failing
        action = initial_state + np.random.normal(0, 0.001, size=8).astype(np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
            
    env.close()
    
    if not frames:
        print("No frames captured.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w, c = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    out.release()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_untrained()
