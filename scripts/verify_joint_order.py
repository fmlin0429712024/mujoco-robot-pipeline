
import sys
import os
import pandas as pd
import numpy as np
import gymnasium as gym

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv

def verify_joint_order():
    print("Verifying Joint Order (Parquet Mode)...")
    
    # 1. Load Parquet
    parquet_path = "local/sim_pick_place_demo/data/chunk-000/file-000.parquet"
    if not os.path.exists(parquet_path):
        print(f"Parquet not found at {parquet_path}")
        return
        
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Columns: {df.columns.tolist()[:10]}...") # Print first 10 columns
    
    # Get first row
    row = df.iloc[0]
    
    # Extract 'observation.state'
    # Check if it's a list column or separate columns
    if "observation.state" in df.columns:
        ds_qpos = np.array(row["observation.state"])
    else:
        # Check for flattened? E.g. observation.state_0
        print("Column 'observation.state' not found. Checking for flattened...")
        state_cols = [c for c in df.columns if "observation.state" in c]
        print(f"State Cols: {state_cols}")
        # Assuming sorted?
        # If flattened, they might be "observation.state_0", "observation.state_1"...
        # or list. Parquet supports list.
        return

    # Check action too
    if "action" in df.columns:
        ds_action = np.array(row["action"])
    else:
        ds_action = None

    print(f"Dataset qpos[0]: {ds_qpos}")
    if ds_action is not None:
        print(f"Dataset action[0]: {ds_action}")
        
    # 2. Reset Environment
    print("\nInitializing Environment...")
    env = TrossenGymEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    
    env_qpos = obs["observation.state"]
    print(f"Environment qpos[0] (Seed 0): {env_qpos}")
    
    # 3. Comparison
    print("\n--- Comparison ---")
    
    diff = np.abs(ds_qpos - env_qpos)
    print(f"Difference: {diff}")
    
    if np.allclose(ds_qpos, env_qpos, atol=1e-3):
        print("SUCCESS: Dataset and Environment start states match.")
    else:
        print("WARNING: Mismatch detected.")
        print(f"Joint 0: DS={ds_qpos[0]:.4f}, Env={env_qpos[0]:.4f}")
        print(f"Joint 1: DS={ds_qpos[1]:.4f}, Env={env_qpos[1]:.4f}")
        print(f"Joint 6: DS={ds_qpos[6]:.4f}, Env={env_qpos[6]:.4f}")

if __name__ == "__main__":
    verify_joint_order()
