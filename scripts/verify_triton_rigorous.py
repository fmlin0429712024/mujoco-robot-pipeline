import time
import numpy as np
import tritonclient.grpc as grpcclient
import sys

def verify_rigorous(url="localhost:8001", model_name="act_pick_place"):
    print(f"Starting rigorous verification for model '{model_name}' at {url}...")
    
    try:
        # Initialize Client
        client = grpcclient.InferenceServerClient(url=url)
        if not client.is_server_ready():
             print("❌ Server is not ready! Exiting.")
             return

        # Prepare Dummy Data
        # Using random data to ensure model processes it (not just caching)
        state = np.random.randn(1, 8).astype(np.float32)
        image = np.random.rand(1, 3, 480, 640).astype(np.float32)

        inputs = [
            grpcclient.InferInput("state__0", state.shape, "FP32"),
            grpcclient.InferInput("image__1", image.shape, "FP32"),
        ]
        inputs[0].set_data_from_numpy(state)
        inputs[1].set_data_from_numpy(image)
        
        outputs = [grpcclient.InferRequestedOutput("output__0")]

        latencies = []
        consistent_shape = True
        all_non_zero = True
        has_nans = False
        first_result_sample = None
        
        print(f"\n🚀 Running 20 consecutive inference calls...")
        
        for i in range(20):
            start_time = time.time()
            response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000 # ms
            latencies.append(latency)
            
            result = response.as_numpy("output__0")
            
            # Capture first result for Detailed Inspection
            if i == 0:
                first_result_sample = result
            
            # 1. Consistency Check: Shape
            if result.shape != (1, 8):
                consistent_shape = False
                print(f"  ❌ Call {i+1}: Shape Mismatch! Got {result.shape}")
            
            # 2. Consistency Check: Non-Zero Data
            # A completely zero array implies serialization failure (the bug we just fixed)
            if np.all(result == 0):
                all_non_zero = False
                print(f"  ❌ Call {i+1}: All Zeros detected!")
            
            # 3. Check for NaNs
            if np.isnan(result).any():
                has_nans = True
                print(f"  ❌ Call {i+1}: NaNs detected!")

            # Simple progress dot
            print(".", end="", flush=True)
            
        print("\n")
            
        # Metrics Calculation
        cold_start = latencies[0]
        warm_avg = np.mean(latencies[1:])
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)
        
        # Report
        print("="*40)
        print("     VERIFICATION REPORT      ")
        print("="*40)
        
        # 1. Consistency
        consistency_pass = consistent_shape and all_non_zero and not has_nans
        print(f"1. CONSISTENCY CHECK: {'✅ PASS' if consistency_pass else '❌ FAIL'}")
        print(f"   - Valid Shape (1, 8):   {'✅ OK' if consistent_shape else '❌ FAIL'}")
        print(f"   - Non-Zero Data:        {'✅ OK' if all_non_zero else '❌ FAIL'}")
        print(f"   - No NaNs:              {'✅ OK' if not has_nans else '❌ FAIL'}")
        
        # 2. Performance
        print(f"\n2. PERFORMANCE METRICS")
        print(f"   - Cold Start Latency:   {cold_start:.2f} ms")
        print(f"   - Warm State Average:   {warm_avg:.2f} ms")
        print(f"   - Min / Max Latency:    {min_lat:.2f} / {max_lat:.2f} ms")
        
        # 3. Numerical Validation
        if first_result_sample is not None:
            vals = first_result_sample.flatten()[:3]
            print(f"\n3. NUMERICAL VALIDATION (Sample Call #1)")
            print(f"   - First 3 Values:       {vals}")
            print(f"   - Data Type:            {first_result_sample.dtype}")
            print(f"   - Memory Layout:        {'C-Contiguous' if first_result_sample.flags['C_CONTIGUOUS'] else 'Non-Contiguous'}")
        
        print("="*40)

        if consistency_pass:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Verification CRASHED with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_rigorous()
