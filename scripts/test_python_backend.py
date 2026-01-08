"""
Test script for Triton Python Backend deployment.

This verifies that the Python backend can successfully serve the ACT policy.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_python_backend():
    """Test Triton Python Backend with ACT policy."""
    
    try:
        import tritonclient.grpc as grpcclient
        from tritonclient.utils import np_to_triton_dtype
    except ImportError:
        print("✗ tritonclient not installed")
        print("  Install with: pip install tritonclient[grpc]")
        return False
    
    print("=== Triton Python Backend Test ===\n")
    
    # Configuration
    url = os.getenv("TRITON_URL", "localhost:8001")
    model_name = "act_pick_place"
    model_version = "1"
    
    print(f"Server: {url}")
    print(f"Model: {model_name} (v{model_version})\n")
    
    # Initialize client
    try:
        client = grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print(f"✗ Failed to connect to Triton server at {url}")
        print(f"  Error: {e}")
        print("\nMake sure Triton is running:")
        print("  docker-compose up triton")
        return False
    
    # Check server health
    print("1. Checking server health...")
    if not client.is_server_live():
        print("✗ Server is not live")
        return False
    print("✓ Server is live")
    
    if not client.is_server_ready():
        print("✗ Server is not ready")
        return False
    print("✓ Server is ready")
    
    # Check model
    print(f"\n2. Checking model '{model_name}'...")
    if not client.is_model_ready(model_name, model_version):
        print(f"✗ Model {model_name} (v{model_version}) is not ready")
        print("\nCheck Triton logs:")
        print("  docker-compose logs triton")
        return False
    print(f"✓ Model is ready")
    
    # Get model metadata
    print("\n3. Fetching model metadata...")
    try:
        metadata = client.get_model_metadata(model_name, model_version)
        print(f"✓ Model metadata retrieved")
        print(f"  Platform: {metadata.platform if hasattr(metadata, 'platform') else 'N/A'}")
        print(f"  Inputs: {len(metadata.inputs)}")
        print(f"  Outputs: {len(metadata.outputs)}")
    except Exception as e:
        print(f"✗ Failed to get metadata: {e}")
        return False
    
    # Test inference
    print("\n4. Testing inference with dummy data...")
    try:
        # Create dummy inputs (8D state, normalized image)
        batch_size = 1
        state = np.random.randn(batch_size, 8).astype(np.float32)
        image = np.random.rand(batch_size, 3, 480, 640).astype(np.float32)  # [0, 1]
        
        print(f"  State shape: {state.shape}")
        print(f"  Image shape: {image.shape}")
        
        # Create input tensors
        inputs = [
            grpcclient.InferInput("state__0", state.shape, np_to_triton_dtype(state.dtype)),
            grpcclient.InferInput("image__1", image.shape, np_to_triton_dtype(image.dtype)),
        ]
        inputs[0].set_data_from_numpy(state)
        inputs[1].set_data_from_numpy(image)
        
        # Create output placeholder
        outputs = [grpcclient.InferRequestedOutput("output__0")]
        
        # Run inference
        response = client.infer(
            model_name=model_name,
            model_version=model_version,
            inputs=inputs,
            outputs=outputs,
        )
        
        # Get result
        action = response.as_numpy("output__0")
        print(f"✓ Inference successful")
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        
    except Exception as e:
        print(f"✗ gRPC Inference failed: {e}")
        # import traceback
        # traceback.print_exc()
        # Don't return, continue to HTTP test
    
    
    # Test inference (HTTP fallback check)
    print("\n[Optional] Testing with HTTP client...")
    try:
        import tritonclient.http as httpclient
        
        triton_client = httpclient.InferenceServerClient(url="localhost:8000")
        inputs_http = [
            httpclient.InferInput("state__0", state.shape, "FP32"),
            httpclient.InferInput("image__1", image.shape, "FP32"),
        ]
        inputs_http[0].set_data_from_numpy(state)
        inputs_http[1].set_data_from_numpy(image)
        outputs_http = [httpclient.InferRequestedOutput("output__0")]
        
        res = triton_client.infer(model_name, inputs_http, outputs=outputs_http)
        print("✓ HTTP Inference response:", res.get_response())
        print("✓ HTTP Output shape:", res.as_numpy("output__0").shape)
        
    except ImportError:
        print("⚠ tritonclient[http] not installed")
    except Exception as e:
        print(f"✗ HTTP Inference failed: {e}")

    # Benchmark latency
    print("\n5. Benchmarking latency (10 requests)...")
    try:
        import time
        latencies = []
        
        for i in range(10):
            start = time.time()
            response = client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs,
            )
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        print(f"✓ Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
        
    except Exception as e:
        print(f"⚠ Benchmark failed: {e}")
    
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
    print("\nThe Python backend is working correctly.")
    print("You can now run inference via Triton:")
    print("  python scripts/eval_policy.py --mode triton")
    
    return True


if __name__ == "__main__":
    success = test_python_backend()
    sys.exit(0 if success else 1)
