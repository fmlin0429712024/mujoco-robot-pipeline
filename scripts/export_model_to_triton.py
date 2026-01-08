"""
Export trained ACT policy to ONNX format for Triton Inference Server.

This script converts a PyTorch ACT policy checkpoint to ONNX format
and places it in the Triton model repository.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lerobot.policies.act.modeling_act import ACTPolicy


def export_to_onnx(
    checkpoint_dir: str,
    output_path: str,
    opset_version: int = 14,
    verify: bool = True,
):
    """
    Export ACT policy to ONNX format.
    
    Args:
        checkpoint_dir: Path to pretrained model directory
        output_path: Output path for ONNX model
        opset_version: ONNX opset version
        verify: Whether to verify exported model
    """
    print(f"Loading policy from {checkpoint_dir}...")
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.eval()
    
    # Move to CPU for export (ONNX export works best on CPU)
    device = torch.device("cpu")
    policy.to(device)
    
    print("Creating dummy inputs...")
    # Create dummy inputs matching expected shapes
    # State: [batch, 8] - 8 joint positions (6 arm + 2 gripper)
    dummy_state = torch.randn(1, 8, dtype=torch.float32, device=device)
    
    # Image: [batch, 3, 480, 640] - ImageNet normalized RGB
    dummy_image = torch.randn(1, 3, 480, 640, dtype=torch.float32, device=device)
    
    dummy_batch = {
        "observation.state": dummy_state,
        "observation.images.top_cam": dummy_image,
    }
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Note: ACT policy's forward() requires 'action' key which we don't have during export
    # We need to use select_action() instead, but that's also not directly exportable
    # Solution: Create a wrapper that only does inference
    
    class ACTInferenceWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, state, image):
            """
            Wrapper for ONNX export.
            
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
            # Use select_action which handles inference without requiring 'action' key
            return self.policy.select_action(batch)
    
    wrapper = ACTInferenceWrapper(policy)
    wrapper.eval()
    
    try:
        # Export to ONNX using wrapper
        torch.onnx.export(
            wrapper,
            (dummy_state, dummy_image),  # Separate args instead of dict
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["observation.state", "observation.images.top_cam"],
            output_names=["action"],
            dynamic_axes={
                "observation.state": {0: "batch_size"},
                "observation.images.top_cam": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
        )
        print(f"✓ Model exported to {output_path}")
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        print("\nTrying TorchScript export as fallback...")
        
        # Fallback to TorchScript
        torchscript_path = output_path.with_suffix(".pt")
        try:
            traced_model = torch.jit.trace(wrapper, (dummy_state, dummy_image))
            traced_model.save(str(torchscript_path))
            print(f"✓ TorchScript model exported to {torchscript_path}")
            print("\nNOTE: Update config.pbtxt to use 'pytorch_libtorch' platform")
            return
        except Exception as ts_error:
            print(f"✗ TorchScript export also failed: {ts_error}")
            raise
    
    if verify:
        print("\nVerifying exported model...")
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model is valid")
            
            # Test inference
            ort_session = ort.InferenceSession(str(output_path))
            
            # Prepare inputs
            ort_inputs = {
                "observation.state": dummy_state.numpy(),
                "observation.images.top_cam": dummy_image.numpy(),
            }
            
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            action_onnx = ort_outputs[0]
            
            # Compare with PyTorch
            with torch.no_grad():
                action_torch = wrapper(dummy_state, dummy_image).cpu().numpy()
            
            # Check if outputs match
            max_diff = np.abs(action_onnx - action_torch).max()
            print(f"✓ Max difference between ONNX and PyTorch: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print("✓ Verification passed! Outputs match.")
            else:
                print(f"⚠ Warning: Outputs differ by {max_diff:.6f}")
                
        except ImportError:
            print("⚠ Skipping verification (onnx/onnxruntime not installed)")
            print("  Install with: pip install onnx onnxruntime")
        except Exception as e:
            print(f"⚠ Verification failed: {e}")
    
    print(f"\n✓ Export complete!")
    print(f"\nNext steps:")
    print(f"1. Verify model file exists: {output_path}")
    print(f"2. Start Triton server: docker-compose up triton")
    print(f"3. Check model status: curl http://localhost:8000/v2/models/act_pick_place")


def main():
    parser = argparse.ArgumentParser(
        description="Export ACT policy to ONNX for Triton Inference Server"
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
        default="model_repository/act_pick_place/1/model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step",
    )
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_dir=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
