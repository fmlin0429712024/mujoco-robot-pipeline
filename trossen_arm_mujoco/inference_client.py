"""
Inference client abstraction for ACT policy.

Supports both Triton Inference Server (gRPC) and local PyTorch inference.
Mode is controlled via INFERENCE_MODE environment variable.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import torch
from pathlib import Path


class InferenceClient(ABC):
    """Abstract base class for inference clients."""
    
    @abstractmethod
    def predict(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run inference on observation and return action.
        
        Args:
            observation: Dictionary containing:
                - "observation.state": Joint positions [8]
                - "observation.images.top_cam": RGB image [H, W, 3]
        
        Returns:
            action: Unnormalized target joint positions [8]
        """
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
    
    @staticmethod
    def create(
        mode: Optional[str] = None,
        triton_url: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> "InferenceClient":
        """
        Factory method to create appropriate inference client.
        
        Args:
            mode: "triton" or "local". Defaults to INFERENCE_MODE env var or "local"
            triton_url: Triton server URL (e.g., "localhost:8001")
            model_name: Model name in Triton repository
            model_version: Model version (default: "1")
            checkpoint_dir: Path to local checkpoint (for local mode)
        
        Returns:
            Configured inference client
        """
        mode = mode or os.getenv("INFERENCE_MODE", "local")
        
        if mode == "triton":
            triton_url = triton_url or os.getenv("TRITON_URL", "localhost:8001")
            model_name = model_name or os.getenv("MODEL_NAME", "act_pick_place")
            model_version = model_version or os.getenv("MODEL_VERSION", "1")
            
            return TritonGRPCClient(
                url=triton_url,
                model_name=model_name,
                model_version=model_version,
            )
        elif mode == "local":
            if checkpoint_dir is None:
                checkpoint_dir = os.getenv(
                    "CHECKPOINT_DIR",
                    "outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model"
                )
            return LocalInferenceClient(checkpoint_dir=checkpoint_dir)
        else:
            raise ValueError(f"Unknown inference mode: {mode}. Use 'triton' or 'local'")


class TritonGRPCClient(InferenceClient):
    """Triton Inference Server client using gRPC protocol."""
    
    def __init__(self, url: str, model_name: str, model_version: str = "1"):
        """
        Initialize Triton gRPC client.
        
        Args:
            url: Triton server gRPC endpoint (e.g., "localhost:8001")
            model_name: Name of model in Triton repository
            model_version: Model version to use
        """
        try:
            import tritonclient.grpc as grpcclient
            from tritonclient.utils import np_to_triton_dtype
        except ImportError:
            raise ImportError(
                "tritonclient not installed. Install with: pip install tritonclient[grpc]"
            )
        
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.grpcclient = grpcclient
        self.np_to_triton_dtype = np_to_triton_dtype
        
        # Initialize client
        self.client = grpcclient.InferenceServerClient(url=url)
        
        # Verify server is live
        if not self.client.is_server_live():
            raise ConnectionError(f"Triton server at {url} is not live")
        
        # Verify model is ready
        if not self.client.is_model_ready(model_name, model_version):
            raise ValueError(f"Model {model_name} version {model_version} is not ready")
        
        print(f"✓ Connected to Triton server at {url}")
        print(f"✓ Model {model_name} (v{model_version}) is ready")
    
    def _preprocess_observation(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess observation for Triton inference.
        
        Applies ImageNet normalization to images and ensures correct shapes (8D state).
        """
        # State: [8] -> [1, 8]
        state = observation["observation.state"].astype(np.float32)
        if state.ndim == 1:
            state = state[np.newaxis, :]
        
        # Image: [H, W, 3] -> [1, 3, H, W] with ImageNet normalization
        image = observation["observation.images.top_cam"].astype(np.float32)
        
        # Handle format (GymEnv returns CHW, but raw images might be HWC)
        if image.shape[0] == 3:
            # Already CHW (3, H, W)
            pass
        elif image.shape[-1] == 3:
            # HWC (H, W, 3) -> Convert to CHW
            image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError(f"Expected image in CHW or HWC format, got shape {image.shape}")
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply ImageNet normalization
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image = (image - imagenet_mean) / imagenet_std
        
        # Add batch dimension
        if image.ndim == 3:
            image = image[np.newaxis, :]  # [1, 3, H, W]
        
        return {
            "observation.state": state,
            "observation.images.top_cam": image,
        }
    
    def predict(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run inference via Triton gRPC.
        
        Args:
            observation: Raw observation from environment
        
        Returns:
            Unnormalized action array [14]
        """
        # Preprocess
        processed = self._preprocess_observation(observation)
        
        # Create input tensors
        inputs = []
        inputs.append(
            self.grpcclient.InferInput(
                "state__0",
                processed["observation.state"].shape,
                self.np_to_triton_dtype(processed["observation.state"].dtype),
            )
        )
        inputs[0].set_data_from_numpy(processed["observation.state"])
        
        inputs.append(
            self.grpcclient.InferInput(
                "image__1",
                processed["observation.images.top_cam"].shape,
                self.np_to_triton_dtype(processed["observation.images.top_cam"].dtype),
            )
        )
        inputs[1].set_data_from_numpy(processed["observation.images.top_cam"])
        
        # Create output placeholder
        outputs = [self.grpcclient.InferRequestedOutput("output__0")]
        
        # Run inference
        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
        )
        
        # Extract action
        action = response.as_numpy("output__0")
        
        # Remove batch dimension: [1, 8] -> [8]
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        
        return action
    
    def close(self):
        """Close gRPC connection."""
        if hasattr(self, "client"):
            self.client.close()


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
