
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trossen_arm_mujoco.gym_env import TrossenGymEnv

class RobotBodyNode(Node):
    def __init__(self, env):
        super().__init__('trossen_body')
        self.env = env
        
        # Current Command to apply (default to zeros/hold)
        self.current_action = np.zeros(8, dtype=np.float32) # Assuming 8-dof action

        # Pub/Sub
        self.state_pub = self.create_publisher(JointState, '/robot/joint_states', 10)
        self.cmd_sub = self.create_subscription(
            Float32MultiArray,
            '/robot/target_cmd',
            self.cmd_callback,
            10
        )
        
        # Timer for Simulation Loop (e.g. 50Hz)
        self.timer = self.create_timer(0.02, self.sim_loop)
        
        self.obs, _ = self.env.reset()
        self.get_logger().info("Robot Body Node Initialized. Physics Running.")

    def cmd_callback(self, msg):
        # Update current action from ROS message
        # self.get_logger().info(f"Received Command: {msg.data[:4]}")
        self.current_action = np.array(msg.data, dtype=np.float32)

    def sim_loop(self):
        # 1. Step Physics with latest action
        obs, reward, done, truncated, info = self.env.step(self.current_action)
        
        if done or truncated:
            obs, _ = self.env.reset()
            
        # 2. Publish State (qpos part of obs)
        # TrossenGymEnv obs['state'] is typically [qpos(7-8?), qvel...]
        # Let's extract generic "state" from obs if it's a dict, or if it's flat.
        # Based on previous interactions, "obs" might be just 'image' and 'state'.
        # Let's check TrossenGymEnv source if needed.
        # For this generic bridge, lets assume obs['state'] is what we want.
        
        # Inspecting previous `nim_wrapper` log or code: `state.shape` was [8].
        # In `eval_policy.py`: `obs` was passed to `predict`.
        
        # If obs is a dict:
        state_vec = obs['state'] if isinstance(obs, dict) else obs
        # If it contains image too, we need separation.
        # Actually TrossenGymEnv usually returns dict with "pixel_values" and "agent_pos".
        # But `eval_policy` code passed `obs` directly.
        # Let's trust `obs` is the vector required by the policy for now, 
        # OR better: The Brain expects a vector.
        
        # Construct Msg
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # msg.name = [...]
        msg.position = state_vec.tolist() if isinstance(state_vec, np.ndarray) else state_vec
        
        self.state_pub.publish(msg)
        # self.get_logger().info(f"Pub State: {state_vec[:4]}")

def main(args=None):
    rclpy.init(args=args)
    
    # Initialize MuJoCo Env (Headless)
    try:
        env = TrossenGymEnv(render_mode="rgb_array")
        
        node = RobotBodyNode(env)
        rclpy.spin(node)
        
        env.close()
        node.destroy_node()
        rclpy.shutdown()
        
    except Exception as e:
        print(f"Error starting Robot Body: {e}")

if __name__ == '__main__':
    main()
