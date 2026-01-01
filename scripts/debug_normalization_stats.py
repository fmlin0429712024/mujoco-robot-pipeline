
import sys
import os
import torch
from safetensors.torch import load_file

def inspect_stats():
    stats_path = "outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors"
    
    if not os.path.exists(stats_path):
        print(f"Stats file not found: {stats_path}")
        return

    print(f"Loading stats from {stats_path}")
    data = load_file(stats_path)
    
    for key, val in data.items():
        if "action" in key:
            print(f"\nKey: {key}")
            print(f"Shape: {val.shape}")
            print(f"Values: {val}")

if __name__ == "__main__":
    inspect_stats()
