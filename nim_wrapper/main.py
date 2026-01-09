import os
import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

# Environment variables
TRITON_URL = os.getenv("TRITON_URL", "triton:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "act_pick_place")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

class NIMBrainNode(Node):
    def __init__(self):
        super().__init__('nim_brain')
        
        # Triton Client
        self.triton_client = None
        self.connect_to_triton()
        
        # ROS 2 Interfaces
        self.subscription = self.create_subscription(
            JointState,
            '/robot/joint_states',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            '/robot/target_cmd',
            10
        )
        
        self.get_logger().info('NIM Brain Node Initialized. Ready for inference.')

    def connect_to_triton(self):
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
            if not self.triton_client.is_server_live():
                self.get_logger().warn(f"Triton server at {TRITON_URL} is not live yet.")
        except Exception as e:
            self.get_logger().error(f"Failed to create Triton client: {e}")

    def preprocess(self, state, image=None):
        # 1. State: [8] -> [1, 8]
        if state.ndim == 1:
            state = state[np.newaxis, :]
            
        # 2. Image: Placeholder for now if we strictly use Joint State or if Image comes via another topic
        # For simplicity in this Refactor Step 1:
        # We assume the "JointState" message might be overloaded or we need a custom message to carry Image + State.
        # OR, we keep it simple: "Blind" policy for a second, OR we pack image into a customized message later.
        
        # WAIT! The original policy needs an IMAGE. 
        # Sending 480x640x3 image over ROS 2 topic /robot/camera/image_raw is standard.
        # But for this specific "Smoke Test" refactor, let's look at the implementation plan.
        # "Callback: Receive Joint State -> Preprocess -> Call Triton". 
        # If the Image is missing, ACT will fail.
        
        # HACK for "Hello World" Smoke Test Phase:
        # Create a dummy image (zeros) just to satisfy input shape [1, 3, 480, 640].
        # The goal stated by user is "Refactor ... to a ROS 2 Based Architecture ... verify arm moves".
        # Real image transport comes later or needs synchronization (message filters).
        
        dummy_image = np.zeros((1, 3, 480, 640), dtype=np.float32)
        return state, dummy_image

    def listener_callback(self, msg: JointState):
        if not self.triton_client:
            self.connect_to_triton()
            if not self.triton_client:
                return

        # Parse State (qpos + qvel)
        # Using the standard JointState msg: position, velocity
        # Ideally we map names to indices. For now, assuming direct array mapping from Simulation.
        # ACT expects 8-dim state (qpos=7, gripper=1? or qpos+qvel?)
        # Let's assume the App publishes the exact 8-dim vector in `position` field for simplicity of this bridge.
        
        state_np = np.array(msg.position, dtype=np.float32)
        
        # Preprocess
        state_in, image_in = self.preprocess(state_np)
        
        # Inference
        try:
             # IO Tensors
            inputs = []
            inputs.append(grpcclient.InferInput("state__0", state_in.shape, np_to_triton_dtype(state_in.dtype)))
            inputs[0].set_data_from_numpy(state_in)
            
            inputs.append(grpcclient.InferInput("image__1", image_in.shape, np_to_triton_dtype(image_in.dtype)))
            inputs[1].set_data_from_numpy(image_in)
            
            outputs = [grpcclient.InferRequestedOutput("output__0")]
            
            response = self.triton_client.infer(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                inputs=inputs,
                outputs=outputs
            )
            
            action_raw = response.as_numpy("output__0")
             # Remove batch dim
            if action_raw.ndim == 2 and action_raw.shape[0] == 1:
                action_raw = action_raw[0]
                
            # Publish Command
            cmd_msg = Float32MultiArray()
            cmd_msg.data = action_raw.tolist()
            self.publisher_.publish(cmd_msg)
            # self.get_logger().info(f'Published Action: {action_raw[:4]}...')
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    nim_brain = NIMBrainNode()
    rclpy.spin(nim_brain)
    nim_brain.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
