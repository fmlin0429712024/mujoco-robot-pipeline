
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image

# Constants
DT = 0.02  # Simulation timestep (check if this matches your recording)


def convert_episode(hdf5_path, output_dir_base):
    filename = os.path.basename(hdf5_path)
    episode_id = filename.replace(".hdf5", "").replace("episode_", "")
    
    # Create image directory for this episode
    img_dir = os.path.join(output_dir_base, "images", f"episode_{episode_id}")
    os.makedirs(img_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, "r") as root:
        # Check available keys
        # Assuming structure:
        # /observations/qpos
        # /observations/images/cam_high (or similar)
        # /action
        
        qpos = root["/observations/qpos"][()]
        actions = root["/action"][()]  # Action taken *at* this step (or resulting in next?)
        # Usually: state[t], action[t] -> state[t+1]
        
        # Images: User asked for 'top_cam'. We'll map 'cam_high' to it.
        # Check which cameras exist
        cam_key = None
        if "cam_high" in root["/observations/images"]:
            cam_key = "cam_high"
        elif "cam_low" in root["/observations/images"]:
            cam_key = "cam_low" # Fallback
        
        if not cam_key:
            print(f"Warning: No suitable camera found in {hdf5_path}. Skipping.")
            return None

        images_np = root[f"/observations/images/{cam_key}"][()]
        
        num_steps = qpos.shape[0]
        # Actions might be 1 less or same length depending on recording.
        # Usually actions match steps if we record (s, a) pairs.
        # record_sim_episodes.py typically records same length.
        
        data_rows = []
        for i in range(num_steps):
            timestamp = i * DT
            
            # Save Image
            # Shape is (H, W, 3) usually? Or (3, H, W)? MuJoCo usually (H, W, 3)
            # visualize_eps.py used: image_dict[cam_name][frame_idx][:, :, [2, 1, 0]] converting to BGR for cv2
            # So stored is likely RGB.
            img_array = images_np[i]
            img = Image.fromarray(img_array)
            img_filename = f"frame_{i:06d}.png"
            img_path_abs = os.path.join(img_dir, img_filename)
            img.save(img_path_abs)
            
            # Relative path for portability (optional, but requested "path")
            # We'll store relative to dataset root
            img_path_rel = os.path.join("images", f"episode_{episode_id}", img_filename)

            row = {
                "observation.state": qpos[i].tolist(),
                "observation.images.top_cam": img_path_rel,
                "action": actions[i].tolist() if i < len(actions) else None,
                "timestamp": timestamp,
                "episode_index": int(episode_id),
                "frame_index": i
            }
            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    return df

def main(args):
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        return

    output_parquet_dir = args.output_dir if args.output_dir else data_dir
    os.makedirs(output_parquet_dir, exist_ok=True)

    hdf5_files = glob.glob(os.path.join(data_dir, "episode_*.hdf5"))
    hdf5_files.sort()
    
    print(f"Found {len(hdf5_files)} HDF5 files to convert.")

    for h5_file in hdf5_files:
        print(f"Converting {h5_file}...")
        try:
            df = convert_episode(h5_file, output_parquet_dir)
            if df is not None:
                # Save as parquet
                filename = os.path.basename(h5_file).replace(".hdf5", ".parquet")
                output_path = os.path.join(output_parquet_dir, filename)
                df.to_parquet(output_path, index=False)
                print(f"Saved {output_path}")
        except Exception as e:
            print(f"Failed to convert {h5_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to Parquet + Images")
    parser.add_argument("--data_dir", required=True, help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", help="Directory to save Parquet files (default: same as data_dir)")
    args = parser.parse_args()
    main(args)
