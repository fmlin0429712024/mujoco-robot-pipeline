
import sys
import os
import site

# Add lerobot scripts to python path to allow importing lerobot_train
lerobot_scripts_path = "/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/lerobot/scripts"
sys.path.append(lerobot_scripts_path)

# Add project root needed for gym env import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Register Environment
import gymnasium as gym
from gymnasium.envs.registration import register
from trossen_arm_mujoco.gym_env import TrossenGymEnv

# Register Environment
if "trossen_arm/OneArmPickPlace-v0" not in gym.envs.registry:
    print("Registering trossen_arm/OneArmPickPlace-v0")
    register(
        id="trossen_arm/OneArmPickPlace-v0",
        entry_point="trossen_arm_mujoco.gym_env:TrossenGymEnv",
        max_episode_steps=600,
    )

# Register Alias for LeRobot gym_manipulator trick
if "gym_gym_manipulator/OneArmPickPlace-v0" not in gym.envs.registry:
    print("Registering alias gym_gym_manipulator/OneArmPickPlace-v0")
    register(
        id="gym_gym_manipulator/OneArmPickPlace-v0",
        entry_point="trossen_arm_mujoco.gym_env:TrossenGymEnv",
        max_episode_steps=600,
    )

# Mock gym_gym_manipulator to allow using env.type=gym_manipulator
import types
gym_manipulator_mock = types.ModuleType("gym_gym_manipulator")
sys.modules["gym_gym_manipulator"] = gym_manipulator_mock

# Import lerobot_train
try:
    import lerobot_train
except ImportError:
    print(f"Could not import lerobot_train from {lerobot_scripts_path}")
    sys.exit(1)

# Construct sys.argv for draccus/argparse
sys.argv = [
    "lerobot_train.py",
    "--policy.type", "act",
    "--dataset.repo_id", "sim_pick_place_demo",
    "--dataset.root", "data/lerobot",
    "--dataset.video_backend", "pyav",
    "--env.type", "gym_manipulator",
    "--env.task", "OneArmPickPlace-v0",
    "--policy.device", "mps",
    "--output_dir", "outputs/train/act_pick_place_30k",
    "--job_name", "act_pick_place_30k",
    "--policy.repo_id", "local/act_pick_place_30k",
    "--policy.push_to_hub", "false",
    "--wandb.enable", "false",
    "--batch_size", "8",
    "--num_workers", "0",
    "--eval.n_episodes", "5",
    "--eval.batch_size", "5",
    "--steps", "30000",
    "--save_freq", "5000",
    "--eval_freq", "31000", 
    "--log_freq", "1000",
    "--policy.pretrained_path", "outputs/train/act_pick_place_10k/checkpoints/010000/pretrained_model",
]

if __name__ == "__main__":
    print(f"Starting training with args: {sys.argv}")
    lerobot_train.main()
