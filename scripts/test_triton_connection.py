"""
Test Triton Inference Server connectivity and model availability.

This script verifies:
1. Triton server is running and accessible
2. Model is loaded and ready
3. Inference works correctly
4. Latency metrics
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_triton_connection(url: str = "localhost:8001", model_name: str = "act_pick_place"):
    """
    Test Triton server connection and model availability.
    
    Args:
        url: Triton gRPC endpoint
        model_name: Model name to test
    """
    print("=== Triton Connection Test ===\n")
    
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("✗ tritonclient not installed")
        print("  Install with: pip install tritonclient[grpc]")
        return False
    
    # Test 1: Server connectivity
    print(f"1. Testing connection to {url}...")
    try:
        client = grpcclient.InferenceServerClient(url=url)
        if client.is_server_live():
            print("   ✓ Server is live")
        else:
            print("   ✗ Server is not live")
            return False
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False
    
    # Test 2: Server health
    print("\n2. Checking server health...")
    try:
        if client.is_server_ready():
            print("   ✓ Server is ready")
        else:
            print("   ✗ Server is not ready")
            return False
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False
    
    # Test 3: Model availability
    print(f"\n3. Checking model '{model_name}'...")
    try:
        if client.is_model_ready(model_name):
            print("   ✓ Model is ready")
        else:
            print("   ✗ Model is not ready")
            return False
    except Exception as e:
        print(f"   ✗ Model check failed: {e}")
        return False
    
    # Test 4: Model metadata
    print("\n4. Fetching model metadata...")
    try:
        metadata = client.get_model_metadata(model_name)
        print(f"   ✓ Model name: {metadata.name}")
        print(f"   ✓ Model versions: {metadata.versions}")
        print(f"   ✓ Platform: {metadata.platform}")
        
        print("\n   Inputs:")
        for inp in metadata.inputs:
            print(f"     - {inp.name}: {inp.datatype} {inp.shape}")
        
        print("\n   Outputs:")
        for out in metadata.outputs:
            print(f"     - {out.name}: {out.datatype} {out.shape}")
            
    except Exception as e:
        print(f"   ✗ Metadata fetch failed: {e}")
        return False
    
    # Test 5: Sample inference
    print("\n5. Testing inference with dummy data...")
    try:
        from tritonclient.utils import np_to_triton_dtype
        
        # Create dummy inputs (8D state for 8 joints)
        dummy_state = np.random.randn(1, 8).astype(np.float32)
        dummy_image = np.random.randn(1, 3, 480, 640).astype(np.float32)
        
        # Create input tensors
        inputs = []
        inputs.append(
            grpcclient.InferInput(
                "observation.state",
                dummy_state.shape,
                np_to_triton_dtype(dummy_state.dtype),
            )
        )
        inputs[0].set_data_from_numpy(dummy_state)
        
        inputs.append(
            grpcclient.InferInput(
                "observation.images.top_cam",
                dummy_image.shape,
                np_to_triton_dtype(dummy_image.dtype),
            )
        )
        inputs[1].set_data_from_numpy(dummy_image)
        
        # Create output placeholder
        outputs = [grpcclient.InferRequestedOutput("action")]
        
        # Run inference with timing
        start_time = time.time()
        response = client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )
        latency = (time.time() - start_time) * 1000  # ms
        
        action = response.as_numpy("action")
        
        print(f"   ✓ Inference successful")
        print(f"   ✓ Latency: {latency:.2f} ms")
        print(f"   ✓ Action shape: {action.shape}")
        print(f"   ✓ Action range: [{action.min():.3f}, {action.max():.3f}]")
        
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Latency benchmark
    print("\n6. Running latency benchmark (10 requests)...")
    try:
        latencies = []
        for _ in range(10):
            start = time.time()
            response = client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
            )
            latencies.append((time.time() - start) * 1000)
        
        print(f"   ✓ Mean latency: {np.mean(latencies):.2f} ms")
        print(f"   ✓ Std latency: {np.std(latencies):.2f} ms")
        print(f"   ✓ Min latency: {np.min(latencies):.2f} ms")
        print(f"   ✓ Max latency: {np.max(latencies):.2f} ms")
        
    except Exception as e:
        print(f"   ✗ Benchmark failed: {e}")
    
    print("\n" + "="*40)
    print("✓ All tests passed!")
    print("="*40)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Triton server connection")
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("TRITON_URL", "localhost:8001"),
        help="Triton gRPC endpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "act_pick_place"),
        help="Model name to test"
    )
    args = parser.parse_args()
    
    success = test_triton_connection(url=args.url, model_name=args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
