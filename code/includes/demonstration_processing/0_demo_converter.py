import os
import pickle
import numpy as np
import transforms3d as t3d
from roboticstoolbox.robot.Robot import Robot
import matplotlib.pyplot as plt

# Get list of data
## Windows path
# rel_path = "C:\\Users\\ven058\\OneDrive - Wageningen University & Research\\"
## Linux path
rel_path = "/media/data/DataRobert/"


class UR5e(Robot):
    """
    Class that imports a UR5e URDF model

    ``UR5e()`` is a class which imports a Universal Robotics UR5e robot
    definition from a URDF file.  The model describes its kinematic and
    graphical characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.UR5()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self, file_path=None):
        
        links, name, urdf_string, urdf_filepath = self.URDF_read(file_path=file_path)
        # for link in links:
        #     print(link)

        super().__init__(
            links,
            name=name.upper(),
            manufacturer="Universal Robotics",
            gripper_links=links[-1],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.qr = np.array([np.pi, 0, 0, 0, np.pi / 2, 0])
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

if rel_path.startswith("C:"):
    robot_file = "C:\\Users\\ven058\\orchard_code\\orchard-reactive-tp-gmr\\lfd_executor\\config\\ur5e.urdf"
else:
    robot_file = "/home/lfd/orchard_ws/src/orchard-reactive-tp-gmr/lfd_executor/config/ur5e.urdf"

ur5e = UR5e(file_path=robot_file)
ur5e_ets = ur5e.ets()

trajectories_path = os.path.join(rel_path, "PhD", "05 Study 4", "04 Robot", "teleop_apple_trajectories")
filelist = os.listdir(trajectories_path)

# Load results
data = {}
for i, file in enumerate(filelist):
    try:
        if file.endswith(".pickle") and file.startswith("teleop"):
            with open(os.path.join(trajectories_path, file), "rb") as f:
                data[file[:-7]] = pickle.load(f)
        print("Loaded file {} of {}".format(i + 1, len(filelist)))
    except:
        print("Failed to load file {}".format(file))

dataset_approach = {}
dataset_retreat = {}
rot_convention = "syzx"
column_reorder = [2, 1, 0]
for i, key in enumerate(data.keys()):
    properties_lst = key.split("_")

    # Get value of each property and put in result of this run
    properties = {}
    for prop in properties_lst:
        parts = prop.split("-")
        if len(parts) == 1:
            continue
        elif len(parts) == 2:
            properties[parts[0]] = parts[1]
        elif len(parts) == 3:
            properties[parts[0]] = parts[1] + "-" + parts[2]

    control_data = data[key][0]
    transform_data = data[key][1]

    # Determine moment vacuum pressure increased
    # Get time vacuum was activated and turned off
    vacuum_on, vacuum_off = control_data[np.where(control_data[:,13] >= 1)[0][[0, -1]], 0]
    start_time = control_data[2,0]

    # Get approach segment
    approach_joints = control_data[np.where(control_data[:,0] >= start_time)[0][0]:np.where(control_data[:,0] >= vacuum_on)[0][0]]
    # Calculate the end effector pose
    approach_pos = []
    approach_euler = []
    approach_quat = []
    for row in approach_joints:
        ee_pose = ur5e_ets.fkine(row[2:8])
        approach_pos.append(ee_pose.t)
        approach_euler.append(t3d.euler.mat2euler(ee_pose.R, axes=rot_convention))
        approach_quat.append(t3d.quaternions.mat2quat(ee_pose.R))
    approach_pos = np.asarray(approach_pos)
    approach_euler = np.asarray(approach_euler)
    # Correct order of RPY
    approach_euler = approach_euler[:,column_reorder]
    approach_quat = np.asarray(approach_quat)

    approach = np.hstack((approach_pos, approach_euler))# , approach_quat))

    # Get retreat segment
    retreat_joints = control_data[np.where(control_data[:,0] >= vacuum_on)[0][0]:np.where(control_data[:,0] >= vacuum_off)[0][0]:]
    # Calculate the end effector pose
    retreat_pos = []
    retreat_euler = []
    retreat_quat = []
    retreat_rot_mat = []
    retreat_trans = []
    for row in retreat_joints:
        ee_pose = ur5e_ets.fkine(row[2:8])
        retreat_pos.append(ee_pose.t)
        retreat_euler.append(t3d.euler.mat2euler(ee_pose.R, axes=rot_convention))
        retreat_quat.append(t3d.quaternions.mat2quat(ee_pose.R))
        retreat_rot_mat.append(ee_pose.R)
        retreat_trans.append(ee_pose)
    retreat_pos = np.asarray(retreat_pos)
    retreat_euler = np.asarray(retreat_euler)
    # Correct order of RPY
    retreat_euler = retreat_euler[:,column_reorder]
    retreat_quat = np.asarray(retreat_quat)
    retreat_rot_mat = np.asarray(retreat_rot_mat)
    
    retreat = np.hstack((retreat_pos, retreat_euler)) # , retreat_quat))

    dataset_approach[key] = approach
    dataset_retreat[key] = retreat    
    
    print("\r Converted demo {} of {}".format(i + 1, len(data.keys())), end="")

# Make all datasets the same length
lengths = [len(dataset_approach[key]) for key in dataset_approach.keys()]
min_length_apprach = int(min(lengths))
lengths = [len(dataset_retreat[key]) for key in dataset_retreat.keys()]
min_length_retreat = int(min(lengths))

for key in dataset_approach.keys():
    # Interpolate approach
    count = len(dataset_approach[key]) / min_length_apprach
    if count == 1:
        interpol = dataset_approach[key]
    else:
        size = (min_length_apprach, dataset_approach[key].shape[1])
        interpol = np.zeros(size)
        for i in range(interpol.shape[0]):
            original_index = i * count
            lower_index = int(np.floor(original_index))
            upper_index = min(int(np.ceil(original_index)),dataset_approach[key].shape[0]-1)
            start = dataset_approach[key][lower_index]
            end = dataset_approach[key][upper_index]
            diff = start - end
            diffstep = original_index - lower_index
            
            interpol[i] = start + diff * diffstep

    dataset_approach[key] = interpol

    # Interpolate retreat
    count = len(dataset_retreat[key]) / min_length_retreat
    if count == 1:
        interpol = dataset_retreat[key]
    else:
        size = (min_length_retreat, dataset_retreat[key].shape[1])
        interpol = np.zeros(size)
        for i in range(interpol.shape[0]):
            original_index = i * count
            lower_index = max(int(np.floor(original_index)),0)
            upper_index = min(int(np.ceil(original_index)),dataset_retreat[key].shape[0]-1)
            start = dataset_retreat[key][lower_index]
            end = dataset_retreat[key][upper_index]
            diff = start - end
            diffstep = original_index - lower_index
            
            interpol[i] = start + diff * diffstep
    dataset_retreat[key] = interpol

# Store the datasets
store_path = os.path.join(rel_path, "PhD", "05 Study 4", "04 Robot", "training_data")
if not os.path.exists(store_path):
    os.makedirs(store_path)

approach_path = os.path.join(store_path, "approach")
if not os.path.exists(approach_path):
    os.makedirs(approach_path)

retreat_path = os.path.join(store_path, "retreat")
if not os.path.exists(retreat_path):
    os.makedirs(retreat_path)

for key in dataset_approach.keys():
    np.savetxt(os.path.join(approach_path, key + ".csv"), dataset_approach[key], delimiter=",")
    np.savetxt(os.path.join(retreat_path, key + ".csv"), dataset_retreat[key], delimiter=",")