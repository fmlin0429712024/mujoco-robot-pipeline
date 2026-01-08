# 08: Triton Inference Architecture (Local Development)

## 🎯 The Big Shift: From Embedded to Client-Server

In this section, we transform our robot's "brain" (the AI model) from being **embedded inside the application** to being a **separate service**.

Think of it like this:
- **Before**: Your app had the AI model built-in (like a calculator app with the math logic inside).
- **After**: Your app talks to a separate AI service (like a banking app talking to a server).

### Visual Comparison

**Before: Embedded Model**
```
┌─────────────────────────────────────────┐
│         Your Application                │
│  ┌───────────────────────────────────┐  │
│  │  Load Model -> Run Prediction     │  │
│  └───────────────────────────────────┘  │
│  Problem: Hard to scale, hard to update │
└─────────────────────────────────────────┘
```

**After: Client-Server Architecture**
```
┌──────────────────────┐         ┌─────────────────────────┐
│   Your Application   │         │   Triton Server         │
│  ┌────────────────┐  │  gRPC   │  ┌───────────────────┐  │
│  │ Inference      │  │────────▶│  │  AI Model         │  │
│  │ Client         │  │ Request │  │  (ACT Policy)     │  │
│  └────────────────┘  │◀────────│  └───────────────────┘  │
└──────────────────────┘         └─────────────────────────┘
```

## 🔑 Key Concepts

### 1. The Inference Client (Abstraction)

We created a wrapper called `InferenceClient` to hide the complexity. Your application code doesn't need to know if the model is running on the same computer or a supercomputer in the cloud.

```python
# Old Way (Complex & Specific)
policy = ACTPolicy.from_pretrained(path)
image = normalize(image)
action = policy.predict(image)

# New Way (Simple & Flexible)
client = InferenceClient.create()  # Auto-detects mode
action = client.predict(obs)       # Just works
```

### 2. The Two Modes

The same code works in two ways, controlled by an environment variable:

*   **1. Local Mode (`INFERENCE_MODE=local`)**:
    *   **Best for:** Development and Debugging.
    *   **How it works:** Loads the PyTorch model directly on your computer (CPU).
    *   **Why use it:** It's simple, requires no server setup, and works offline.

*   **2. Triton Mode (`INFERENCE_MODE=triton`)**:
    *   **Best for:** Production and Cloud.
    *   **How it works:** Sends data to a Triton Inference Server (Local Docker or Cloud).
    *   **Why use it:** It scales, handles multiple apps, and can use powerful remote GPUs.

## 🛠️ Hands-On: Running Locally

Since we are still developing, we primarily use **Local Mode**. This lets us verify our logic without needing complex server infrastructure.

### 1. Configure for Local Mode

Simply set the environment variable:

```bash
export INFERENCE_MODE=local
```

### 2. Run the Evaluation

```bash
python scripts/eval_policy.py --mode local --episodes 5
```

You will see it loading the checkpoint directly. This confirms that your code is "Triton-Ready" (using the client abstraction) even though it's running locally.

## What's Next?

Now that our code is architected correctly, we are ready to deploy the "Server" side of the equation. In the next chapter, we will deploy this to a powerful **Google Cloud L4 Instance** to unlock GPU acceleration.
