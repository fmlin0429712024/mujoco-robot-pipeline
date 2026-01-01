"""
Visualize expert/scripted policy doing pick and place (Phase 1 demo).
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.ee_sim_env import OneArmPickPlaceEETask
from trossen_arm_mujoco.scripted_policy import PickAndPlacePolicy
from trossen_arm_mujoco.utils import make_sim_env

def visualize_expert(output_path="visualizations/expert_demo.mp4", episode_len=500):
    env = make_sim_env(
        task_class=OneArmPickPlaceEETask,
        xml_file="trossen_one_arm_scene.xml",
        task_name="sim_pick_place",
        onscreen_render=False,
        cam_list=["cam_high"],
        random=True,
    )
    
    np.random.seed(42)
    ts = env.reset()
    policy = PickAndPlacePolicy(inject_noise=False)
    
    frames = []
    
    for step in range(episode_len):
        # Capture frame
        img = ts.observation["images"]["cam_high"]
        frames.append(img)
        
        # Get action from expert policy
        action = policy(ts)
        ts = env.step(action)
        
        if ts.last():
            # Capture final frames
            for _ in range(30):
                frames.append(ts.observation["images"]["cam_high"])
            break
    
    if not frames:
        print("No frames captured.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w, c = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    out.release()
    print(f"Saved visualization to {output_path}")
    print(f"Final reward: {ts.reward}")

if __name__ == "__main__":
    visualize_expert()
