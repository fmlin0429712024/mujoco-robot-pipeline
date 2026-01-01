
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from trossen_arm_mujoco.sim_env import OneArmPickPlaceTask
from trossen_arm_mujoco.utils import make_sim_env

class TrossenGymEnv(gym.Env):
    """
    Gymnasium wrapper for Trossen Arm MuJoCo environment (Joint Control).
    Compatible with LeRobot.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}


    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        # Use make_sim_env to standardise creation (hooks up physics, etc)
        # We use OneArmPickPlaceTask (Joint Control) because policy outputs joint positions.
        self.env = make_sim_env(
            task_class=OneArmPickPlaceTask,
            xml_file="trossen_one_arm_scene_joint.xml",
            task_name="sim_pick_place",
            onscreen_render=(render_mode == "human"),
            random=True # Use randomization during Eval
        )
        
        # Action Space: 8 joints (6 arm + 2 gripper actuators)
        # Limits: -pi to pi is safe generic, or check specifics. 
        # Using float32 is important for LeRobot/PyTorch.
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(8,), dtype=np.float32
        )
        
        # Observation Space:
        # LeRobot usually uses features dict.
        # "observation.state": 8 joints
        # "observation.images.top_cam": (3, 480, 640)
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            ),
            "observation.images.top_cam": spaces.Box(
                low=0, high=255, shape=(3, 480, 640), dtype=np.uint8
            )
        })
        
        self.task = "Pick up the red cube and place it in the green bucket."
        self.task_description = self.task

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             # If make_sim_env / task supports seeding via method?
             # OneArmPickPlaceTask uses global np.random or similar?
             # We should seed numpy if possible.
             np.random.seed(seed)
             
        ts = self.env.reset()
        return self._format_obs(ts), {}

    def step(self, action):
        ts = self.env.step(action)
        
        # LeRobot expects (obs, reward, terminated, truncated, info)
        terminated = ts.last() # Success or fail?
        # dm_control ts.last() is typically 'end of episode'.
        # We treat it as terminated.
        truncated = False
        
        reward = ts.reward if ts.reward is not None else 0.0
        
        return self._format_obs(ts), reward, terminated, truncated, {}

    def _format_obs(self, ts):
        # Extract qpos (8-dim for OneArmPickPlaceTask)
        qpos = ts.observation["qpos"].astype(np.float32)
        
        # Extract Image (H, W, C) -> (C, H, W)
        img = ts.observation["images"]["cam_high"]
        img_chw = np.moveaxis(img, -1, 0)
        
        obs = {
            "observation.state": qpos,
            "observation.images.top_cam": img_chw
        }
        
        return obs

    def render(self):
        # dm_control handles onscreen rendering if enabled.
        # If rgb_array requested:
        if self.render_mode == "rgb_array":
             return self.env.physics.render(height=480, width=640, camera_id="cam_high")
    
    def close(self):
        self.env.close()
