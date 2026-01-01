
import sys
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

def check_policy_methods():
    checkpoint_dir = "outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model"
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    print("Policy Methods:")
    print([m for m in dir(policy) if "stats" in m or "norm" in m])
    
    # Also check if we can access the NormalizerProcessor
    # usually policy.mapper or something?

if __name__ == "__main__":
    check_policy_methods()
