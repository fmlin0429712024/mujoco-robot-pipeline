# NIM-Style Microservices Architecture

## Overview

We have evolved our inference stack from a direct client-server model to a **NIM-style (NVIDIA Inference Microservice)** architecture. This introduces a dedicated intermediate layer—the **NIM Wrapper**—between the application logic and the high-performance inference engine (Triton).

### Why this architecture?

1.  **Decoupling at the Protocol Level**: The Client Application (`trossen_arm_mujoco`) no longer needs to import complex libraries like `tritonclient` or manage gRPC channels. It speaks pure, standard `JSON/HTTP`.
2.  **Logic Encapsulation**: Special preprocessing (cleaning outputs, reshaping tensors) lives "near the model" in the wrapper, not "near the robot". This makes the robot code cleaner and the model more portable.
3.  **Future-Proofing (Security & Auth)**: This wrapper is the perfect place to enforce API Keys, OAuth, or Rate Limiting before requests ever hit the heavy inference engine.

## Architecture

```mermaid
graph LR
    subgraph "Robot / Client App"
        Client[NIM Client] 
        Logic[Control Policy]
    end

    subgraph "NIM Microservice Layer"
        API[FastAPI Wrapper]
        Auth[Security / Auth (Future)]
        Pre[Preprocessing]
        Post[Postprocessing]
    end

    subgraph "Inference Backend"
        Triton[Triton Server]
        Model[ACT Model (ONNX)]
    end

    Logic --> Client
    Client -- "JSON (HTTP POST)" --> API
    API --> Auth
    Auth --> Pre
    Pre -- "Tensors (gRPC)" --> Triton
    Triton --> Model
    Model --> Triton
    Triton -- "Tensors (gRPC)" --> Post
    Post --> API
    API -- "JSON (Action)" --> Client
```

## Implementation Details

### 1. The Wrapper Service (`nim_wrapper/`)
A lightweight Python container running `FastAPI` and `Uvicorn`.
-   **Endpoint**: `POST /predict`
-   **Input**: JSON with `state` (list) and `image` (nested list).
-   **Output**: JSON with `action` (list).
-   **Responsibilities**:
    -   Validates generic JSON inputs.
    -   Converts data to NumPy arrays and Triton-friendly tensors.
    -   Handles the efficient gRPC connection to Triton.
    -   Exposes a `/health` endpoint for readiness checks.

### 2. The Client (`NIMClient`)
A "dumb" HTTP client located in `trossen_arm_mujoco/inference_client.py`.
-   Uses standard `urllib` (no external heavy dependencies).
-   Treats the AI brain as a black box: "Here is what I see (Observation), tell me what to do (Action)."

### 3. Deployment
The wrapper is deployed as a sidecar or standalone service in `docker-compose.yml`.

```yaml
  nim-wrapper:
    build:
      context: ./nim_wrapper
    environment:
      - TRITON_URL=triton:8001
      - MODEL_NAME=act_pick_place
    depends_on:
      triton:
        condition: service_healthy
```

## Future Security & Extensibility

Users often ask: *"Where do I add API Keys?"* or *"How do I secure this?"*

**The NIM Wrapper is the correct place for this.**

Because Triton is designed for raw performance inside a trusted network, it doesn't natively handle complex keys or user management easily. You should:
1.  **Add Middleware to FastAPI**: In `nim_wrapper/main.py`, simply add a dependency that expects an `Authorization: Bearer <token>` header.
2.  **Keep Triton Private**: In a real production cluster, verify that Triton's ports (8000-8002) are *not* exposed to the public internet, while the NIM Wrapper's port *is* (protected by your auth logic).
