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

results_df = pd.read_csv(rel_path + 'results.csv')

# Calculate pose RPY
results_df["fruit_roll"], results_df["fruit_pitch"], results_df["fruit_yaw"] = zip(*results_df.apply(lambda row: quat_to_euler(row["fruit_quat_w"], row["fruit_quat_x"], row["fruit_quat_y"], row["fruit_quat_z"]), axis=1))

# Get unique poses
results_df = results_df.drop_duplicates(subset=["fruit_pos_x", "fruit_pos_y", "fruit_pos_z", "fruit_quat_x", "fruit_quat_y", "fruit_quat_z", "fruit_quat_w"])

# Testing all parameters
testing_dict = {
    "apple": [5, 10, 20, 40],
    "pear": [5, 10, 20, 40],
}

try: 
    for fruit_type in testing_dict.keys():
        for num_demos in testing_dict[fruit_type]:
            # Loop over apple poses in results_df
            for index, row in results_df.iterrows():
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
                        "--qx="+str(row.fruit_quat_x),
                        "--qy="+str(row.fruit_quat_y),
                        "--qz="+str(row.fruit_quat_z),
                        "--qw="+str(row.fruit_quat_w),
                    ],
                )
except KeyboardInterrupt:
    print("Keyboard interrupt detected, stopping the process")
    pass

end_time = time.time()
duration = end_time - start_time
print("Duration: " + str(duration))
