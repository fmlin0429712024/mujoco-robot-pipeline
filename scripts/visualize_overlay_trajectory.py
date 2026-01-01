
import gymnasium as gym
import torch
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import collections

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.gym_env import TrossenGymEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file

def visualize_trajectory_overlay(ckpt_path, output_video="visualizations/policy_intent.mp4"):
    print(f"Loading policy from {ckpt_path}...")
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.eval()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy.to(device)

    # Load stats
    stats_path = os.path.join(ckpt_path, "policy_preprocessor_step_3_normalizer_processor.safetensors")
    if os.path.exists(stats_path):
        stats = load_file(stats_path)
        action_mean = stats["action.mean"].to(device)
        action_std = stats["action.std"].to(device)
        print("Loaded normalization stats.")
    else:
        print("WARNING: Stats not found!")
        return

    env = TrossenGymEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    
    frames = []
    
    # History buffers
    history_len = 200
    j1_history = collections.deque([obs["observation.state"][1]] * history_len, maxlen=history_len)
    j2_history = collections.deque([obs["observation.state"][2]] * history_len, maxlen=history_len)
    gripper_history = collections.deque([obs["observation.state"][6]] * history_len, maxlen=history_len)
    
    fig, axs = plt.subplots(3, 1, figsize=(4, 6))
    plt.tight_layout()
    
    print("Running Episode...")
    max_steps = 200 # Shorten for debug
    
    success = False
    
    for step in range(max_steps):
        # Prepare Input
        state = torch.from_numpy(obs["observation.state"].copy()).float().unsqueeze(0).to(device)
        image_raw = torch.from_numpy(obs["observation.images.top_cam"].copy()).float()
        image = (image_raw / 255.0).unsqueeze(0).to(device) # Manual image norm
        
        batch = {
            "observation.state": state,
            "observation.images.top_cam": image
        }
        
            # Inference
        with torch.no_grad():
            # Returns [B, Chunk, D]
            action_enc = policy.predict_action_chunk(batch)
            
            print(f"Action Chunk Shape: {action_enc.shape}")

            
        # Unnorm
        action_unnorm = action_enc * action_std + action_mean
        action_np = action_unnorm.squeeze(0).cpu().numpy() # [100, 8]?
        
        # ACT predicts [Chunk_Size, Action_Dim]. usually [100, 8]
        # We take the first action for control
        current_action = action_np[0]
        
        # Step Env
        obs, reward, terminated, truncated, info = env.step(current_action)
        
        # Update History
        curr_qpos = obs["observation.state"]
        j1_history.append(curr_qpos[1])
        j2_history.append(curr_qpos[2])
        gripper_history.append(curr_qpos[6])
        
        # == Visualization ==
        # Render Camera
        cam_frame = env.render() # [H, W, 3]
        
        # Render Plots
        for ax in axs: ax.clear()
        
        # Plot J1 (Shoulder)
        axs[0].plot(range(len(j1_history)), j1_history, 'k-', label="Actual")
        # Plot Prediction (Future)
        pred_j1 = action_np[:, 1] # 100 steps
        axs[0].plot(range(len(j1_history)-1, len(j1_history)-1 + len(pred_j1)), pred_j1, 'r--', label="Pred")
        axs[0].set_ylabel("J1 (Shoulder)")
        axs[0].legend(loc='upper left', fontsize='small')
        
        # Plot J2 (Elbow)
        axs[1].plot(range(len(j2_history)), j2_history, 'k-', label="Actual")
        pred_j2 = action_np[:, 2]
        axs[1].plot(range(len(j2_history)-1, len(j2_history)-1 + len(pred_j2)), pred_j2, 'r--', label="Pred")
        axs[1].set_ylabel("J2 (Elbow)")
        
        # Plot Gripper
        axs[2].plot(range(len(gripper_history)), gripper_history, 'k-', label="Actual")
        pred_grip = action_np[:, 6]
        axs[2].plot(range(len(gripper_history)-1, len(gripper_history)-1 + len(pred_grip)), pred_grip, 'r--', label="Pred")
        axs[2].set_ylabel("Gripper")
        
        # Draw canvas
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        # Modern matplotlib
        # tostring_rgb is gone. Use buffer_rgba
        buf = np.asarray(canvas.buffer_rgba())
        plot_img = buf[:, :, :3] # RGBA -> RGB
        
        # Resize plot to match cam height?
        # Resize plot to match cam height?
        h, w, _ = cam_frame.shape
        ph, pw, _ = plot_img.shape
        
        # Resize plot to height of cam
        scale = h / ph
        plot_resized = cv2.resize(plot_img, (int(pw * scale), h))
        
        combined = np.hstack([cam_frame, plot_resized])
        frames.append(combined)

        if reward == 4:
            success = True
            print("SUCCESS!")
            break
            
    # Save Video
    if frames:
        print(f"Saving video to {output_video}...")
        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print("Done.")

if __name__ == "__main__":
    ckpt = "outputs/train/act_pick_place_30k/checkpoints/030000/pretrained_model"
    visualize_trajectory_overlay(ckpt)
