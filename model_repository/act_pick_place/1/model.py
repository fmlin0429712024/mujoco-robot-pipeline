"""
Triton Python Backend for ACT Policy Inference.

This model runs inside the Triton container and handles stateful ACT policy inference.
"""

import json
import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils
from pathlib import Path


class TritonPythonModel:
    """
    Triton Python Backend implementation for ACT policy.
    
    This allows serving the stateful ACT model without ONNX/TorchScript export.
    """

    def initialize(self, args):
        """
        Initialize the model. Called once when Triton loads the model.
        
        Args:
            args: Dictionary with model configuration
        """
        # Parse model config
        self.model_config = json.loads(args['model_config'])
        
        # Get model instance directory
        self.model_instance_dir = args['model_instance_device_id']
        
        # Import ACT policy (requires project code to be in PYTHONPATH)
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                f"Failed to import required modules. "
                f"Ensure project code is mounted and in PYTHONPATH: {e}"
            )
        
        # Load checkpoint path from environment or use default
        import os
        checkpoint_dir = os.getenv(
            "CHECKPOINT_DIR",
            "/workspace/outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model"
        )
        
        print(f"[Triton Python Backend] Loading ACT policy from: {checkpoint_dir}")
        
        # Load policy
        self.policy = ACTPolicy.from_pretrained(checkpoint_dir)
        self.policy.eval()
        
        # Determine device (CPU for Mac, GPU for Linux)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"[Triton Python Backend] Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            # macOS Metal Performance Shaders (not recommended for Triton)
            self.device = torch.device("cpu")
            print("[Triton Python Backend] Using CPU (MPS not supported in Docker)")
        else:
            self.device = torch.device("cpu")
            print("[Triton Python Backend] Using CPU")
        
        self.policy.to(self.device)
        
        # Load normalization stats
        stats_path = Path(checkpoint_dir) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        self.action_mean = None
        self.action_std = None
        
        if stats_path.exists():
            print(f"[Triton Python Backend] Loading normalization stats from: {stats_path}")
            stats = load_file(str(stats_path))
            self.action_mean = stats["action.mean"].to(self.device)
            self.action_std = stats["action.std"].to(self.device)
        else:
            print("[Triton Python Backend] WARNING: Stats file not found. Using unnormalized output.")
        
        # ImageNet normalization constants
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        
        print("[Triton Python Backend] Model initialized successfully")

    def execute(self, requests):
        """
        Execute inference for a batch of requests.
        
        Args:
            requests: List of pb_utils.InferenceRequest
            
        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []
        
        for request in requests:
            try:
                # Get input tensors
                state_tensor = pb_utils.get_input_tensor_by_name(request, "state__0")
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image__1")
                
                # Convert to numpy
                state_np = state_tensor.as_numpy()  # [batch, 8]
                image_np = image_tensor.as_numpy()  # [batch, 3, 480, 640]
                
                # Preprocess and run inference (returns pb_utils.Tensor)
                output_tensor = self._predict(state_np, image_np)
                
                # Create response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)
                
            except Exception as e:
                # Return error response
                error_message = f"Inference failed: {str(e)}"
                print(f"[Triton Python Backend] ERROR: {error_message}")
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_message)
                )
                responses.append(inference_response)
        
        return responses

    def _predict(self, state_np, image_np):
        """
        Run inference on preprocessed inputs.
        
        Args:
            state_np: State array [batch, 8]
            image_np: Image array [batch, 3, 480, 640] (already in CHW, normalized [0-1])
            
        Returns:
            action_np: Action array [batch, 8]
        """
        # print(f"[Triton Python Backend] _predict called with state shape: {state_np.shape}, image shape: {image_np.shape}")
        
        with torch.no_grad():
            # Convert to torch tensors
            state = torch.from_numpy(state_np).float().to(self.device)
            image = torch.from_numpy(image_np).float().to(self.device)
            
            # print(f"[Triton Python Backend] Converted to torch - state: {state.shape}, image: {image.shape}")
            
            # Apply ImageNet normalization to image
            image = (image - self.imagenet_mean) / self.imagenet_std
            
            # Create batch dictionary
            batch = {
                "observation.state": state,
                "observation.images.top_cam": image,
            }
            
            # print(f"[Triton Python Backend] Calling policy.select_action...")
            
            # Run inference
            action = self.policy.select_action(batch)
            
            # print(f"[Triton Python Backend] policy.select_action returned: {action.shape if hasattr(action, 'shape') else type(action)}")
            # print(f"[Triton Python Backend] action content: {action}")
            
            # Unnormalize action if stats available
            if self.action_mean is not None:
                action = action * self.action_std + self.action_mean
                # print(f"[Triton Python Backend] After unnormalization: {action}")
            
            # DLPack Zero-Copy Transfer
            # 1. Detach and move to CPU (Triton Python Backend usually expects CPU tensors for response unless configured otherwise)
            # 2. Ensure contiguous and float32
            # 3. Create Triton Tensor via DLPack
            
            action = action.detach().cpu().float()
            if not action.is_contiguous():
                action = action.contiguous()
                
            # print(f"[Triton Python Backend] action tensor before dlpack: shape={action.shape}, dtype={action.dtype}, device={action.device}")
            
            # Create output tensor using DLPack
            output_tensor = pb_utils.Tensor.from_dlpack("output__0", to_dlpack(action))
            
            return output_tensor

    def finalize(self):
        """
        Clean up resources. Called when Triton unloads the model.
        """
        print("[Triton Python Backend] Finalizing model...")
        # PyTorch models don't need explicit cleanup
        pass
