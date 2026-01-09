import os
import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="ACT Policy NIM Wrapper")

# Environment variables
TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "act_pick_place")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# Global client
triton_client: Optional[grpcclient.InferenceServerClient] = None

class ObservationRequest(BaseModel):
    state: List[float]
    image: List[float]  # Flattened or structured, we'll assume flattened for simplicity or check structured
    # Actually, for JSON, nested lists are standard for images [H, W, 3] or [3, H, W]

class ActionResponse(BaseModel):
    action: List[float]

def get_client():
    global triton_client
    if triton_client is None:
        try:
            triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
        except Exception as e:
            print(f"Failed to create Triton client: {e}")
            return None
    return triton_client

@app.get("/health")
def health():
    client = get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Triton client not initialized")
    
    try:
        if not client.is_server_live():
             raise HTTPException(status_code=503, detail="Triton server not live")
        if not client.is_model_ready(MODEL_NAME, MODEL_VERSION):
             raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "healthy", "triton": "live", "model": "ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess(state_list: List[float], image_list: List[float]):
    # Convert lists to numpy
    state = np.array(state_list, dtype=np.float32)
    # Assume image is passed as a flat list or nested list. Let's support nested [H, W, 3] from client
    # But wait, passing large images via JSON is inefficient. 
    # For this architecture demonstration, we'll accept it.
    image = np.array(image_list, dtype=np.float32)

    # 1. State: [8] -> [1, 8]
    if state.ndim == 1:
        state = state[np.newaxis, :]
    
    # 2. Image Processing
    # Logic copied from InferenceClient
    
    # GymEnv usually returns CHW or HWC.
    # We expect the client to send what it has.
    # If client sends nested list [[...], [...]] it will be N-D array.
    
    # Handle HWC -> CHW conversion
    if image.shape[0] == 3:
        pass # Already CHW
    elif image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1)) # HWC -> CHW
    
    # Normalize [0, 1]
    image = image / 255.0
    
    # ImageNet Standard
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = (image - imagenet_mean) / imagenet_std
    
    # Add batch dim [1, 3, H, W]
    if image.ndim == 3:
        image = image[np.newaxis, :]
        
    return state, image

@app.post("/predict", response_model=ActionResponse)
def predict(request: ObservationRequest):
    client = get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Triton unavailable")

    try:
        # Preprocess
        # Note: In a real high-perf scenario, client might send base64 or bytes
        # Here we accept standard JSON lists for simplicity/compatibility
        state, image = preprocess(request.state, request.image)
        
        # IO Tensors
        inputs = []
        inputs.append(grpcclient.InferInput("state__0", state.shape, np_to_triton_dtype(state.dtype)))
        inputs[0].set_data_from_numpy(state)
        
        inputs.append(grpcclient.InferInput("image__1", image.shape, np_to_triton_dtype(image.dtype)))
        inputs[1].set_data_from_numpy(image)
        
        outputs = [grpcclient.InferRequestedOutput("output__0")]
        
        # Inference
        response = client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=inputs,
            outputs=outputs
        )
        
        action = response.as_numpy("output__0")
        
        # Remove batch dim
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
            
        return {"action": action.tolist()}

    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
