
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.inference_client import InferenceClient


def check_inference_client():
    """
    Debug inference client setup and connectivity.
    """
    print("=== Inference Client Debug ===\n")
    
    # Check environment
    inference_mode = os.getenv("INFERENCE_MODE", "local")
    print(f"INFERENCE_MODE: {inference_mode}")
    
    if inference_mode == "triton":
        triton_url = os.getenv("TRITON_URL", "localhost:8001")
        model_name = os.getenv("MODEL_NAME", "act_pick_place")
        model_version = os.getenv("MODEL_VERSION", "1")
        print(f"TRITON_URL: {triton_url}")
        print(f"MODEL_NAME: {model_name}")
        print(f"MODEL_VERSION: {model_version}")
    else:
        checkpoint_dir = os.getenv(
            "CHECKPOINT_DIR",
            "outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model"
        )
        print(f"CHECKPOINT_DIR: {checkpoint_dir}")
    
    print("\nInitializing client...")
    try:
        client = InferenceClient.create()
        print("✓ Client initialized successfully")
        
        # Test with dummy observation (8D state for 8 joints)
        import numpy as np
        dummy_obs = {
            "observation.state": np.random.randn(8).astype(np.float32),
            "observation.images.top_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        
        print("\nTesting inference with dummy observation...")
        action = client.predict(dummy_obs)
        print(f"✓ Inference successful")
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        client.close()
        print("\n✓ All checks passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_inference_client()
