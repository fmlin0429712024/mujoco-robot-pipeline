
import sys
import os
import numpy as np
from dm_control import mujoco

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trossen_arm_mujoco.sim_env import OneArmPickPlaceTask
from trossen_arm_mujoco.utils import make_sim_env
from trossen_arm_mujoco.constants import ASSETS_DIR

def check_joint_order():
    print("Checking Joint Order in Simulation...")
    
    # Load the joint-control XML directly to inspect model
    xml_path = os.path.join(ASSETS_DIR, "trossen_one_arm_scene_joint.xml")
    if not os.path.exists(xml_path):
        print(f"Error: XML not found at {xml_path}")
        return

    physics = mujoco.Physics.from_xml_path(xml_path)
    
    print(f"Total qpos dim: {physics.model.nq}")
    print(f"Total qvel dim: {physics.model.nv}")
    
    print("\n--- Joint Names (qpos indices) ---")
    # Iterate through joints
    # qpos address might be different from joint id if using free joints (7 dof)
    # But here we likely have hinge/slide joints (1 dof)
    
    qpos_idx = 0
    for i in range(physics.model.njnt):
        name = physics.model.id2name(i, 'joint')
        jnt_type = physics.model.jnt_type[i]
        qpos_width = 7 if jnt_type == 0 else 1 # 0 is free joint
        
        print(f"Joint ID {i}: '{name}' (Type: {jnt_type}, qpos_idx: {qpos_idx}:{qpos_idx+qpos_width})")
        qpos_idx += qpos_width

    print("\n--- Sim Env Logic Check ---")
    # Check what OneArmPickPlaceTask slices
    # It takes [:8]
    print("OneArmPickPlaceTask takes qpos[:8]. Based on above, does this cover [joint_0...gripper_r]?")

    # Load Env
    env = make_sim_env(
        task_class=OneArmPickPlaceTask,
        xml_file="trossen_one_arm_scene_joint.xml",
        task_name="sim_pick_place",
        onscreen_render=False
    )
    
    ts = env.reset()
    obs_qpos = ts.observation["qpos"]
    print(f"\nEnv Reset Observation 'qpos' shape: {obs_qpos.shape}")
    print(f"Values: {obs_qpos}")
    
    print("\n--- Dataset Sample Comparison ---")
    # Quick check of dataset first frame
    try:
        import h5py
        # find a file
        data_dir = os.path.join(os.path.dirname(__file__), "../local/sim_pick_place_demo")
        files = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
        if files:
            with h5py.File(os.path.join(data_dir, files[0]), 'r') as f:
                ds_qpos = f["observations/qpos"][0]
                print(f"Dataset Episode {files[0]} Frame 0 qpos: {ds_qpos}")
                print(f"Range check: Env vs Dataset")
                print(f"Env Max: {obs_qpos.max()}, Min: {obs_qpos.min()}")
                print(f"DS  Max: {ds_qpos.max()}, Min: {ds_qpos.min()}")
    except ImportError:
        print("h5py not installed, skipping dataset check")
    except Exception as e:
        print(f"Could not check dataset: {e}")

if __name__ == "__main__":
    check_joint_order()
