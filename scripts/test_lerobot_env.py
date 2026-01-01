
import gymnasium as gym
import numpy as np
import torch
import sys
import os

# Add project root needed for gym env import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv
from gymnasium.vector import SyncVectorEnv

def make_env():
    return TrossenGymEnv()

def test_wrapper():
    print("Testing TrossenGymEnv direct...")
    env = TrossenGymEnv()
    obs, info = env.reset(seed=42)
    print("Direct reset keys:", obs.keys())
    print("Direct state shape:", obs["observation.state"].shape)
    
    print("\nTesting VectorEnv...")
    vec_env = SyncVectorEnv([make_env for _ in range(2)])
    obs, info = vec_env.reset(seed=42)
    print("Vector reset keys:", obs.keys())
    if "observation.state" in obs:
        print("Vector state shape:", obs["observation.state"].shape)
    else:
        print("Keys missing in VectorEnv!")

    # Check for LeRobot GymWrapper if available
    try:
        from lerobot.envs.gym import GymWrapper
        from lerobot.envs.configs import EnvConfig
        print("\nTesting LeRobot GymWrapper...")
        
        cfg = EnvConfig(name="trossen_arm/OneArmPickPlace-v0", task="OneArmPickPlace-v0")
        # Wrapper needs an instantiated env usually, or we pass args?
        # Actually GymWrapper takes (env, cfg).
        
        le_env = GymWrapper(env, cfg)
        obs, info = le_env.reset(seed=42)
        print("LeRobot Wrapper reset keys:", obs.keys())
    except Exception as e:
        print(f"\nLeRobot Wrapper test failed: {e}")

if __name__ == "__main__":
    test_wrapper()
