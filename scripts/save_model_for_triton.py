"""
Save ACT policy in PyTorch format for Triton PyTorch backend.

This script saves the policy model in a format compatible with Triton's
pytorch_libtorch backend, avoiding the complexity of ONNX export.
"""

import argparse
import sys
import os
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lerobot.policies.act.modeling_act import ACTPolicy


def save_for_triton_pytorch(
    checkpoint_dir: str,
    output_path: str,
):
    """
    Save ACT policy for Triton PyTorch backend.
    
    Args:
        checkpoint_dir: Path to pretrained model directory
        output_path: Output path for model file
    """
    print(f"Loading policy from {checkpoint_dir}...")
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.eval()
    
    # Move to CPU for serving
    device = torch.device("cpu")
    policy.to(device)
    
    print("Creating inference wrapper...")
    
    # Create wrapper that Triton can use
    class ACTInferenceWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, state, image):
            """
            Inference method for Triton.
            
            Args:
                state: [batch, 8] joint positions
                image: [batch, 3, 480, 640] ImageNet normalized image
            
            Returns:
                action: [batch, 8] predicted actions
            """
            batch = {
                "observation.state": state,
                "observation.images.top_cam": image,
            }
            return self.policy.select_action(batch)
    
    wrapper = ACTInferenceWrapper(policy)
    wrapper.eval()
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {output_path}...")
    
    # Save using TorchScript scripting (not tracing)
    try:
        # Use torch.jit.script for models with control flow
        scripted_model = torch.jit.script(wrapper)
        scripted_model.save(str(output_path))
        print(f"✓ Model saved to {output_path}")
    except Exception as e:
        print(f"✗ Scripting failed: {e}")
        print("\nSaving as regular PyTorch model...")
        # Fallback: save the entire wrapper
        torch.save(wrapper, str(output_path))
        print(f"✓ Model saved to {output_path} (PyTorch format)")
    
    print(f"\n✓ Export complete!")
    print(f"\nNext steps:")
    print(f"1. Update config.pbtxt platform to 'pytorch_libtorch'")
    print(f"2. Start Triton server: docker-compose up triton")
    print(f"3. Check model status: curl http://localhost:8000/v2/models/act_pick_place")


def main():
    parser = argparse.ArgumentParser(
        description="Save ACT policy for Triton PyTorch backend"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model",
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_repository/act_pick_place/1/model.pt",
        help="Output path for model file",
    )
    
    args = parser.parse_args()
    
    save_for_triton_pytorch(
        checkpoint_dir=args.checkpoint,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
