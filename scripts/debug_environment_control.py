
import sys
import os
import numpy as np
import imageio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv

def debug_control():
    print("Testing Environment Control...")
    env = TrossenGymEnv(render_mode="rgb_array")
    
    obs, _ = env.reset()
    start_qpos = obs["observation.state"]
    print(f"Start qpos: {start_qpos}")
    
    # Target: Move J1 to 1.5 rad (approx 85 deg)
    target_action = start_qpos.copy()
    target_action[1] = 1.5 
    
    print(f"Target action: {target_action}")
    
    frames = []
    
    for i in range(50):
        # Step env with constant action
        obs, reward, terminated, truncated, info = env.step(target_action)
        frames.append(env.render())
        
        if i % 10 == 0:
            print(f"Step {i}: qpos={obs['observation.state']}")
            
    final_qpos = obs["observation.state"]
    print(f"Final qpos: {final_qpos}")
    
    diff = final_qpos[1] - start_qpos[1]
    print(f"J1 Change: {diff}")
    
    if abs(diff) > 0.1:
        print("SUCCESS: Robot moved.")
    else:
        print("FAILURE: Robot did not move.")
        
    # Save video
    imageio.mimsave("visualizations/debug_control.mp4", frames, fps=30)
    print("Saved visualizations/debug_control.mp4")

if __name__ == "__main__":
    debug_control()
