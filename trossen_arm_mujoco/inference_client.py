import os
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import torch
from pathlib import Path


class InferenceClient(ABC):
    """Abstract base class for inference clients."""
    
    @abstractmethod
    def predict(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Run inference on observation and return action."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
    
    @staticmethod
    def create(
        mode: Optional[str] = None,
        api_url: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        # Legacy params ignored in NIM mode but kept for compat
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> "InferenceClient":
        """
        Factory method to create appropriate inference client.
        
        Args:
            mode: "nim" or "local". Defaults to INFERENCE_MODE env var or "local"
            api_url: URL of the NIM wrapper (e.g. "http://nim-wrapper:8000")
        """
        mode = mode or os.getenv("INFERENCE_MODE", "local")
        
        if mode == "nim":
            api_url = api_url or os.getenv("INFERENCE_API_URL", "http://localhost:8090")
            print(f"Initializing NIM Client connecting to {api_url}")
            return NIMClient(url=api_url)
            
        elif mode == "local":
            if checkpoint_dir is None:
                checkpoint_dir = os.getenv(
                    "CHECKPOINT_DIR",
                    "outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model"
                )
            return LocalInferenceClient(checkpoint_dir=checkpoint_dir)
            
        else:
            raise ValueError(f"Unknown inference mode: {mode}. Use 'nim' or 'local'")


class NIMClient(InferenceClient):
    """
    NIM Microservice Client.
    Communicates via JSON/REST with the decoupled wrapper service.
    Zero knowledge of Triton/gRPC/Tensors.
    """
    
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.predict_endpoint = f"{self.url}/predict"
        self.health_endpoint = f"{self.url}/health"
        
        # Verify connection
        try:
            with urllib.request.urlopen(self.health_endpoint) as response:
                if response.status != 200:
                    raise ConnectionError(f"NIM Health check failed: {response.status}")
                print(f"✓ Connected to NIM Wrapper at {self.url}")
        except urllib.error.URLError as e:
            print(f"WARNING: Could not connect to NIM Wrapper at {self.url}: {e}")
            print("Ensure the 'nim-wrapper' service is running.")

    def predict(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Send specific observation to NIM wrapper via generic JSON API.
        """
        # Prepare payload
        # 1. State: numpy -> list
        state = observation["observation.state"].tolist()
        
        # 2. Image: numpy -> list
        # Check format. Wrapper expects standard nested list.
        image = observation["observation.images.top_cam"]
        if hasattr(image, 'tolist'):
            image = image.tolist()
            
        payload = {
            "state": state,
            "image": image
        }
        
        # Send Request
        req = urllib.request.Request(
            self.predict_endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                action = np.array(result["action"], dtype=np.float32)
                return action
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"NIM Inference failed: {e.code} {e.reason}")
        except Exception as e:
            raise RuntimeError(f"NIM Request failed: {str(e)}")

    def close(self):
        pass


class LocalInferenceClient(InferenceClient):
    """Local PyTorch inference client (fallback for development)."""
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize local inference client.
        
        Args:
            checkpoint_dir: Path to pretrained model directory
        """
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "lerobot not installed. Install with: pip install lerobot"
            )
        
        self.checkpoint_dir = checkpoint_dir
        
        # Load policy
        print(f"Loading policy from {checkpoint_dir}...")
        self.policy = ACTPolicy.from_pretrained(checkpoint_dir)
        self.policy.eval()
        
        # Determine device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy.to(self.device)
        
        # Load normalization stats
        stats_path = Path(checkpoint_dir) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        self.action_mean = None
        self.action_std = None
        
        if stats_path.exists():
            print(f"Loading normalization stats from {stats_path}")
            stats = load_file(str(stats_path))
            self.action_mean = stats["action.mean"].to(self.device)
            self.action_std = stats["action.std"].to(self.device)
        else:
            print("WARNING: Stats file not found. Assuming unnormalized output.")
        
        print("✓ Local inference client initialized")
    
    def predict(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run local PyTorch inference.
        
        Args:
            observation: Raw observation from environment
        
        Returns:
            Unnormalized action array [14]
        """
        # Prepare state
        state = torch.from_numpy(observation["observation.state"].copy()).float()
        state = state.unsqueeze(0).to(self.device)  # [1, 8]
        
        # Prepare image
        image_raw = observation["observation.images.top_cam"]
        
        # Handle both numpy and torch tensors
        if hasattr(image_raw, 'numpy'):
            image_raw = image_raw.numpy()
        
        # Check format and convert if needed
        # Gym env returns CHW [3, H, W], we need to handle that
        if image_raw.shape[0] == 3:
            # Already in CHW format, convert to HWC for processing
            image_hwc = np.transpose(image_raw, (1, 2, 0))  # [3, H, W] -> [H, W, 3]
        elif image_raw.shape[-1] == 3:
            # Already in HWC format
            image_hwc = image_raw
        else:
            raise ValueError(f"Unexpected image shape: {image_raw.shape}")
        
        # Convert to torch and to CHW for model
        image = torch.from_numpy(image_hwc.copy()).float()
        image = image.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply ImageNet normalization
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - imagenet_mean) / imagenet_std
        
        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # Create batch
        batch = {
            "observation.state": state,
            "observation.images.top_cam": image,
        }
        
        # Run inference
        with torch.no_grad():
            action = self.policy.select_action(batch)
        
        # Unnormalize if stats available
        if self.action_mean is not None:
            action = action * self.action_std + self.action_mean
        
        # Convert to numpy: [1, 8] -> [8]
        action_np = action.squeeze(0).cpu().numpy()
        
        return action_np
    
    def close(self):
        """Clean up resources."""
        # PyTorch models don't need explicit cleanup
        pass
