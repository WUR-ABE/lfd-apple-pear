import subprocess
import signal
import time
import transforms3d as t3d
import os
import pandas as pd
import psutil
import numpy as np

def quat_to_euler(quaternion_w, quaternion_x, quaternion_y, quaternion_z, axes="syzx", reorder=[2, 1, 0]):
   if reorder is not None:
      angles = t3d.euler.quat2euler((quaternion_w, quaternion_x, quaternion_y, quaternion_z), axes=axes)
      roll = angles[reorder[0]]
      pitch = angles[reorder[1]]
      yaw = angles[reorder[2]]
   else:
      roll, pitch, yaw = t3d.euler.quat2euler((quaternion_w, quaternion_x, quaternion_y, quaternion_z), axes=axes)
   
   return roll, pitch, yaw

start_time = time.time()

# Load goal poses from results file
rel_path = '/media/data/DataRobert/PhD/05 Study 4/04 Robot/analysis_files/'

all_test_poses_dfs = {
    "ZY_flat": {
        "apple": pd.read_csv(rel_path + 'ZY_apple_test_poses_flat.csv'),
        "pear": pd.read_csv(rel_path + 'ZY_pear_test_poses_flat.csv')},
    "XY_ori_centered": {
        "apple": pd.read_csv(rel_path + 'XY_apple_test_poses_ori_centered.csv'),
        "pear": pd.read_csv(rel_path + 'XY_pear_test_poses_ori_centered.csv')},       
}

# Testing all parameters
testing_dict = {
    "apple": [5, 10, 20, 40],
    "pear": [5, 10, 20, 40],
}

try:
    for sweeping_test in all_test_poses_dfs.keys():
        print("Starting sweeping test: " + sweeping_test)
        test_poses_dfs = all_test_poses_dfs[sweeping_test]
        
        # Load results file
        try:
            results_df = pd.read_csv("/media/data/DataRobert/PhD/05 Study 4/04 Robot/sim_sweep_trajectories/" + sweeping_test + "_test_results.csv")
        except FileNotFoundError:
            results_df = pd.DataFrame(columns=["fruit_type", "number_demonstrations", "fruit_pos_x", "fruit_pos_y", "fruit_pos_z", "fruit_roll", "fruit_pitch", "fruit_yaw", "success"])

        for fruit_type in testing_dict.keys():
            for num_demos in testing_dict[fruit_type]:
                # Loop over fruit poses in results_df
                poses_df = test_poses_dfs[fruit_type]

                # Get results for this fruit type and number of demonstrations
                test_results_df = results_df[(results_df.fruit_type == fruit_type) & (results_df.number_demonstrations == num_demos)]

                # Remove poses in poses_df that have already been tested
                for index, row in test_results_df.iterrows():
                    poses_df = poses_df[~((np.abs(poses_df.fruit_pos_x - row.fruit_pos_x) < 1e-5) & (np.abs(poses_df.fruit_pos_y - row.fruit_pos_y) < 1e-5) & (np.abs(poses_df.fruit_pos_z - row.fruit_pos_z) < 1e-5))]

                print("Testing fruit type: " + fruit_type + ", number of demonstrations: " + str(num_demos))
                print("Number of poses to test: " + str(len(poses_df)))

                for index, row in poses_df.iterrows():
                    # Convert to quaternion
                    quat = t3d.euler.euler2quat(row.fruit_yaw, row.fruit_pitch, row.fruit_roll, axes="syzx")

                    while psutil.virtual_memory().percent > 50:
                        print("\rMemory usage is {0}, waiting for it to drop below 50%".format(psutil.virtual_memory().percent), end="")
                        time.sleep(1.0)
                    subprocess.run(
                        [
                            "python3",
                            "../../lfd_executor/lfd_executor/simulation_pick_executor.py",
                            "--fruit_type="+fruit_type,
                            "--num_demos="+str(num_demos),
                            "--tx="+str(row.fruit_pos_x),
                            "--ty="+str(row.fruit_pos_y),
                            "--tz="+str(row.fruit_pos_z),
                            "--qx="+str(quat[1]),
                            "--qy="+str(quat[2]),
                            "--qz="+str(quat[3]),
                            "--qw="+str(quat[0]),
                            "--write_df=True",
                            "--df_file_name="+str(sweeping_test) + "_test_results.csv",
                        ],
                    )
except KeyboardInterrupt:
    print("Keyboard interrupt detected, stopping the process")
    pass

end_time = time.time()
duration = end_time - start_time
print("Duration: " + str(duration))
