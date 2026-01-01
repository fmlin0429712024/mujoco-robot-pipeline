
import sys
import os
import pandas as pd
import numpy as np

def check_diff():
    parquet_path = "local/sim_pick_place_demo/data/chunk-000/file-000.parquet"
    if not os.path.exists(parquet_path):
        print("Parquet not found.")
        return
        
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Extract
    states = np.array(df["observation.state"].tolist())
    actions = np.array(df["action"].tolist())
    
    diff = np.abs(actions - states)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max Diff (Action - State): {max_diff}")
    print(f"Mean Diff: {mean_diff}")
    
    # Check if action[t] == state[t+1]?
    # Shift actions
    # action[t] should predict state[t+1] approx?
    # But action is the command.
    
    # Check if completely identical
    if max_diff < 1e-6:
        print("CRITICAL: Action is IDENTICAL to State. The dataset is recording current position as action!")
        print("The policy will learn to stay still.")
    else:
        print("Action differs from state. (Good)")
        
        # Check magnitude
        print(f"Avg Action Step Size: {mean_diff}")

if __name__ == "__main__":
    check_diff()
