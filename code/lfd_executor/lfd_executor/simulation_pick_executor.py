# Import modules
import numpy as np
import pickle
import transforms3d as t3d
import pbdlib as pbd
import sys
import getopt
import time
from copy import deepcopy
import pandas as pd
import os
import time

# Rel import
from pbdlib_custom import gmmr

# Fix the change in structure between pickle and ROS2 structure
sys.modules["gmmr"] = gmmr

class SimPickExecutor(object):

    def __init__(self, sys_argv=None):
        self.apple_tx = -0.942376315
        self.apple_ty = 0.01365872
        self.apple_tz = 0.737165217
        self.apple_qw = 0.520938681
        self.apple_qx = -0.380781646
        self.apple_qy = -0.472710585
        self.apple_qz = 0.600144092
        self.fruit_type = "apple"
        self.num_demos = 40
        self.write_df = False
        self.df_file_name = "test_results.csv"


        try:
            opts, args = getopt.getopt(sys_argv,"h",["tx=", 
                                                     "ty=", 
                                                     "tz=", 
                                                     "qx=", 
                                                     "qy=", 
                                                     "qz=", 
                                                     "qw=", 
                                                     "fruit_type=", 
                                                     "num_demos=",
                                                     "write_df=",
                                                     "df_file_name=",
                                                     ])
        except getopt.GetoptError:
            print ('ERROR: test.py -i <inputfile> -o <outputfile>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print ('test.py -i <inputfile> -o <outputfile>')
                sys.exit()
            elif opt == "--tx":
                self.apple_tx = float(arg)
            elif opt == "--ty":
                self.apple_ty = float(arg)
            elif opt == "--tz":
                self.apple_tz = float(arg)
            elif opt == "--qx":
                self.apple_qx = float(arg)
            elif opt == "--qy":
                self.apple_qy = float(arg)
            elif opt == "--qz":
                self.apple_qz = float(arg)
            elif opt == "--qw":
                self.apple_qw = float(arg)
            elif opt == "--fruit_type":
                self.fruit_type = arg
            elif opt == "--num_demos":
                self.num_demos = int(arg)
            elif opt == "--write_df":
                self.write_df = bool(arg)
            elif opt == "--df_file_name":
                self.df_file_name = arg

        # Set up model name
        approach_folder = "/media/data/DataRobert/PhD/05 Study 4/04 Robot/test_setup/{0}_{1}_demos/model/approach/".format(self.fruit_type, self.num_demos)
        placing_folder = "/media/data/DataRobert/PhD/05 Study 4/04 Robot/test_setup/{0}_{1}_demos/model/placing/".format(self.fruit_type, self.num_demos)

        # List models in folder
        approach_models = os.listdir(approach_folder)
        placing_models = os.listdir(placing_folder)

        self.model_name = approach_folder + approach_models[-1]
        self.placing_model_name = placing_folder + placing_models[-1]

        # Determine rotation convention and column reorder
        if self.fruit_type == "apple" and self.num_demos == 5:
            self.rot_convention = "sxyz"
            self.column_reorder = [0, 1, 2]
        else:
            self.rot_convention = "syzx"
            self.column_reorder = [2, 1, 0]

        # Helper variables
        self.publish_rate = 120.0
        self.controller_rate = 120.0
        self.controller_type = "DualLQR"
        self.data_folder = "/media/data/DataRobert/PhD/05 Study 4/04 Robot/sim_sweep_trajectories/"
        self.reg_factor = -1.5
        self.transforms = []

        # Set up frames
        self.get_relaxed_frame()
        self.get_start_frame()
        self.get_base_link_frame()
        self.get_place_frame()
        self.get_apple_from_manipulator()

        self.final_fractions = [
            0.5,
            0.45,
            0.4,
            0.35,
            0.3,
            0.25,
            0.2,
            0.15,
            0.1,
            0.05,
        ]

        df_columns = [
            "fruit_type",
            "number_demonstrations",
            "action_type",
            "time_stamp",
            "fruit_pos_x",
            "fruit_pos_y",
            "fruit_pos_z",
            "fruit_quat_w",
            "fruit_quat_x",
            "fruit_quat_y",
            "fruit_quat_z",
            "placement_pos_x",
            "placement_pos_y",
            "placement_pos_z",
            "placement_quat_w",
            "placement_quat_x",
            "placement_quat_y",
            "placement_quat_z",
            "approach_time",
            "total_time",
            "placing_time",
            "max_ori_change",
            "mae_x_0.075",
            "mae_y_0.075",
            "mae_z_0.075",
            "mae_roll_0.075",
            "mae_pitch_0.075",
            "mae_yaw_0.075",
            "num_close_points_0.075",
            "final_offset_x",
            "final_offset_y",
            "final_offset_z",
            "final_offset_roll",
            "final_offset_pitch",
            "final_offset_yaw",
        ]

        for final_fraction in self.final_fractions:
            df_columns += [
                "mae_x_fraction_"+str(final_fraction),
                "mae_y_fraction_"+str(final_fraction),
                "mae_z_fraction_"+str(final_fraction),
                "mae_roll_fraction_"+str(final_fraction),
                "mae_pitch_fraction_"+str(final_fraction),
                "mae_yaw_fraction_"+str(final_fraction),
                "num_close_points_fraction_"+str(final_fraction),
            ]

        # Check if csv file exists
        if not os.path.exists(self.data_folder + self.df_file_name):
            # Create csv file
            df = pd.DataFrame(columns=df_columns)
            df.to_csv(self.data_folder + self.df_file_name, index=False)


        self.intialize_control()

    def get_start_frame(self):
        # Set transformation matrix
        tx = -0.440
        ty = 0.167
        tz = 0.565
        rx = -0.5
        ry = -0.5
        rz = 0.5
        rw = 0.5

        tf_relaxed_to_init_manip = np.eye(4)
        tf_relaxed_to_init_manip[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
        tf_relaxed_to_init_manip[0:3, 3] = np.asarray((tx, ty, tz))

        self.tf_relaxed_to_init_manip = tf_relaxed_to_init_manip

    def get_relaxed_frame(self):
        tf_base_link_to_relaxed = np.eye(4)
        tf_base_link_to_relaxed[0:3, 3] = np.asarray((0.2, 0.0, 0.0))

        self.tf_base_link_to_relaxed = tf_base_link_to_relaxed
        self.tf_base_link_to_relaxed_grasp = deepcopy(tf_base_link_to_relaxed)

    def get_base_link_frame(self):
        # Set transformation matrix
        tx = 0.0
        ty = 0.0
        tz = 0.0
        rx = 0.0
        ry = 0.0
        rz = 1.0
        rw = 0.0

        tf_base_to_base_link = np.eye(4)
        tf_base_to_base_link[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
        tf_base_to_base_link[0:3, 3] = np.asarray((tx, ty, tz))

        self.tf_base_to_base_link = tf_base_to_base_link

    def get_apple_from_manipulator(self):
        # Apple pose from orchard, TODO: more options for sim testing
        tx = self.apple_tx
        ty = self.apple_ty
        tz = self.apple_tz
        qx = self.apple_qx
        qy = self.apple_qy
        qz = self.apple_qz
        qw = self.apple_qw

        tf_relaxed_to_apple = np.eye(4)
        tf_relaxed_to_apple[0:3, 3] = np.asarray((tx, ty, tz))
        tf_relaxed_to_apple[0:3, 0:3] = t3d.quaternions.quat2mat((qw, qx, qy, qz))

        self.tf_base_link_to_apple = np.linalg.multi_dot([self.tf_base_link_to_relaxed, tf_relaxed_to_apple])

    def get_place_frame(self):
        # Set transformation matrix
        tx = 0.125
        ty = 0.425
        tz = 0.255
        rx = -0.912
        ry = 0.0
        rz = 0.0
        rw = 0.411

        tf_world_to_place = np.eye(4)
        tf_world_to_place[0:3, 3] = np.asarray((tx, ty, tz))
        tf_world_to_place[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
        self.tf_base_link_to_place = tf_world_to_place

    def matrix_to_pose(self, matrix, axes="sxyz", reorder=None):
        x, y, z = matrix[0:3, 3]
        if reorder is not None:
            angles = t3d.euler.mat2euler(matrix[0:3, 0:3], axes=axes)
            roll = angles[reorder[0]]
            pitch = angles[reorder[1]]
            yaw = angles[reorder[2]]
        else:
            roll, pitch, yaw = t3d.euler.mat2euler(matrix[0:3, 0:3], axes=axes)
        pose = np.asarray((x, y, z, roll, pitch, yaw))
        return pose

    def transform_control(self, control, matrix):
        # Transform position vector to robot base frame
        pos_vec = control[0:3].reshape((3, 1))
        pos_tvec = np.dot(matrix[0:3, 0:3], pos_vec)

        # Transform orientation vector to robot base frame
        rot_vec = control[3:6].reshape((3, 1))
        rot_tvec = np.dot(matrix[0:3, 0:3], rot_vec)

        # Combine position and orientation
        transformed_control = np.vstack((pos_tvec, rot_tvec)).flatten()
        return transformed_control

    def array_to_matrix(self, vector, axes="sxyz", reorder=None):
        # Get tranformation matrix from array
        orientation = vector[3:6]
        if reorder is not None:
            orientation = orientation[reorder]
        rot_array = t3d.euler.euler2mat(orientation[0], orientation[1], orientation[2], axes=axes)
        tf = np.eye(4)
        tf[0:3, 0:3] = rot_array
        tf[0:3, 3] = vector[0:3]
        return tf

    def transform_logger(self):
        # Log transform between base and dynamic frame
        try:
            tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(self.rtde_r.getActualQ()))
            # Calculate matrix between ee and moving frame when apple was grasped
            tf_relaxed_frame_to_ee = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), tf_base_to_tcp])
            quaternion_relaxed_frame_to_ee = t3d.quaternions.mat2quat(tf_relaxed_frame_to_ee[0:3, 0:3])
            self.transforms += [
                [
                    time.time(),
                    tf_relaxed_frame_to_ee[0, 3],
                    tf_relaxed_frame_to_ee[1, 3],
                    tf_relaxed_frame_to_ee[2, 3],
                    quaternion_relaxed_frame_to_ee[1],
                    quaternion_relaxed_frame_to_ee[2],
                    quaternion_relaxed_frame_to_ee[3],
                    quaternion_relaxed_frame_to_ee[0],
                ]
            ]
        except:
            pass

    def timer_callback(self, time_step):
        # Integer time step
        current_time = time.time()
        t = int(time_step / 100 * (self.lqr_end.horizon-2))

        if t > self.lqr_end.horizon - 2:
            print("Time step too large")  
            t = self.lqr_end.horizon - 2

        # Get current state
        self.xis += [self.xi_g[-1][:-1]] # [pose_i] # 

        if self.controller_type == "DualLQR":
            # Get current state in start frame
            pose_i_start = deepcopy(self.xis[-1])

            # Determine both control inputs
            self.us_start += [-self.lqr_start._K[t].dot(pose_i_start) + self.lqr_start._Kv[t].dot(self.lqr_start._v[t + 1])]
            self.us_end += [-self.lqr_end._K[t].dot(self.xis[-1]) + self.lqr_end._Kv[t].dot(self.lqr_end._v[t + 1])]

            # Get rotation matrix between robot base frame and relaxed_ik frame
            transformed_us_start = self.us_start[-1]

            # Get both inverse covariance matrices
            lambda_start, _ = self.lqr_start.get_Q_z(t)
            lambda_end, _ = self.lqr_end.get_Q_z(t)

            # Make everything but diagonal zero
            lambda_start = np.diag(np.diag(lambda_start))
            lambda_end = np.diag(np.diag(lambda_end))

            # Compute product of both control inputs
            weighted_us = np.linalg.inv(lambda_start + lambda_end).dot(lambda_start.dot(transformed_us_start) + lambda_end.dot(self.us_end[-1]))
            self.us += [weighted_us]
        else:
            raise ValueError("Controller type not supported")

        # Goal pose, based on current pose and control input based on current pose
        # Add time fraction to goal pose
        self.xi_g += [
            np.append(
                self.lqr_end.A.dot(self.xis[-1]) + self.lqr_end.B.dot(self.us[-1]),
                time_step,
            )
        ]  

        # Store controls
        self.controls += [
            np.hstack((current_time, time.time(), self.xis[-1], self.us[-1], self.xi_g[-1]))
        ]
    
    def timer_callback_placing(self, time_step):
        # Integer time step
        current_time = time.time()
        t = int(time_step / 100 * (self.lqr_end_placing.horizon-2))

        if t > self.lqr_end_placing.horizon - 2:
            print("Time step too large")  
            t = self.lqr_end_placing.horizon - 2

        # Get current state
        self.xis += [self.xi_g[-1][:-1]] # [pose_i] # 

        if self.controller_type == "DualLQR":
            # Get current state in start frame
            pose_i_start = deepcopy(self.xis[-1])

            # Determine both control inputs
            self.us_start += [-self.lqr_start_placing._K[t].dot(pose_i_start) + self.lqr_start_placing._Kv[t].dot(self.lqr_start_placing._v[t + 1])]
            self.us_end += [-self.lqr_end_placing._K[t].dot(self.xis[-1]) + self.lqr_end_placing._Kv[t].dot(self.lqr_end_placing._v[t + 1])]

            # Get rotation matrix between robot base frame and relaxed_ik frame
            tf_moving_frame_to_relaxed_frame = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed_grasp), self.tf_base_link_to_relaxed])
            transformed_us_start = self.transform_control(self.us_start[-1], tf_moving_frame_to_relaxed_frame)

            # Get both inverse covariance matrices
            lambda_start, _ = self.lqr_start_placing.get_Q_z(t)
            lambda_end, _ = self.lqr_end_placing.get_Q_z(t)

            # Make everything but diagonal zero
            lambda_start = np.diag(np.diag(lambda_start))
            lambda_end = np.diag(np.diag(lambda_end))

            # Compute product of both control inputs
            weighted_us = np.linalg.inv(lambda_start + lambda_end).dot(lambda_start.dot(transformed_us_start) + lambda_end.dot(self.us_end[-1]))
            self.us += [weighted_us]
        else:
            raise ValueError("Controller type not supported")

        # Goal pose, based on current pose and control input based on current pose
        # Add time fraction to goal pose
        self.xi_g += [
            np.append(
                self.lqr_end_placing.A.dot(self.xis[-1]) + self.lqr_end_placing.B.dot(self.us[-1]),
                time_step,
            )
        ]

        # Store controls
        self.controls += [
            np.hstack((current_time, time.time(), self.xis[-1], self.us[-1], self.xi_g[-1]))
        ]

    def intialize_control(self):
        try:
            # Set success boolean
            success = True

            
            # Get tf between ee and relaxed_frame
            tf_relaxed_frame_to_ee = self.tf_relaxed_to_init_manip
            # Calculate pose
            pose_0 = self.matrix_to_pose(tf_relaxed_frame_to_ee, axes=self.rot_convention, reorder=self.column_reorder)
            # print("Pose 0: {0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f} {5:0.3f}".format(pose_0[0], pose_0[1], pose_0[2], pose_0[3], pose_0[4], pose_0[5]))

            self.initial_pose = pose_0

            # Get transform between base and apple
            tf_relaxed_frame_to_apple = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), self.tf_base_link_to_apple])
            pose_apple = self.matrix_to_pose(tf_relaxed_frame_to_apple, axes=self.rot_convention, reorder=self.column_reorder)
            # print("Pose apple: {0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f} {5:0.3f}".format(pose_apple[0], pose_apple[1], pose_apple[2], pose_apple[3], pose_apple[4], pose_apple[5]))

            # Get tf between place and moving frame, going through grasp_cube
            tf_moving_frame_to_place = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), self.tf_base_link_to_place])
            pose_place = self.matrix_to_pose(tf_moving_frame_to_place, axes=self.rot_convention, reorder=self.column_reorder)
            # print("Pose place: {0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f} {5:0.3f}".format(pose_place[0], pose_place[1], pose_place[2], pose_place[3], pose_place[4], pose_place[5]))
            
            # Load model from file location
            self.model = pickle.load(open(self.model_name, "rb"), encoding="latin1")
            self.placing_model = pickle.load(open(self.placing_model_name, "rb"), encoding="latin1")

            # Predict trajectory (outside of model to also get the sigma)
            # Create arrays for transforming data
            A0 = np.identity(n=7)
            An = np.identity(n=7)
            Ap = np.identity(n=7)
            b0 = np.zeros(7)
            bn = np.zeros(7)
            bp = np.zeros(7)
            A0[1:7, 1:7], b0[1:7] = pbd.utils.inv_for_lintrans(pose_0)
            An[1:7, 1:7], bn[1:7] = pbd.utils.inv_for_lintrans(pose_apple)
            Ap[1:7, 1:7], bp[1:7] = pbd.utils.inv_for_lintrans(pose_place)

            # Get time at 125 Hz
            length = int(self.publish_rate * (len(self.model.t) / self.controller_rate))
            self.time_axis = np.linspace(0, 100, length)

            place_length = int(self.publish_rate * (len(self.placing_model.t) / self.controller_rate))
            self.place_time_axis = np.linspace(0, 100, place_length)

            # Set required variables for LQR
            A, b = pbd.utils.get_canonical(6, 1, 1.0 / self.publish_rate)
            self.sq = [i for i in range(0, len(self.time_axis))]
            self.place_sq = [i for i in range(0, len(self.place_time_axis))]

            # Select columns for split models
            self.dim1 = np.array([0, 1, 2, 3, 4, 5, 6])
            self.dim2 = np.array([0, 7, 8, 9, 10, 11, 12])

            # Put in initial pose
            self.xis = [pose_0]

            # Initialize first control input
            self.us = [np.zeros(6)]

            # List of goal states, initial goal is start pose
            # Add time to start pose
            init_goal = np.append(pose_0, 0)
            self.xi_g = [init_goal]

            # Set up LQR based on controller_type
            if self.controller_type == "DualLQR":
                # Split models
                _mod1 = self.model.gmm_.marginal_array(self.dim1).lintrans(A0, b0)
                _mod2 = self.model.gmm_.marginal_array(self.dim2).lintrans(An, bn)

                # Get the most probable trajectory and uncertainty
                mu1, sigma1 = _mod1.condition(self.time_axis[:, None], dim_in=slice(0, 1), dim_out=slice(0, 7))
                mu2, sigma2 = _mod2.condition(self.time_axis[:, None], dim_in=slice(0, 1), dim_out=slice(0, 7))

                # Set up LQR
                self.lqr_start = pbd.LQR(A, b, dt=1.0 / self.publish_rate, horizon=len(mu1))
                self.lqr_start.gmm_xi = [mu1[:, 1:], sigma1[:, 1:, 1:], self.sq]
                self.lqr_start.gmm_u = self.reg_factor

                self.lqr_end = pbd.LQR(A, b, dt=1.0 / self.publish_rate, horizon=len(mu2))
                self.lqr_end.gmm_xi = [mu2[:, 1:], sigma2[:, 1:, 1:], self.sq]
                self.lqr_end.gmm_u = self.reg_factor

                # Fit LQR
                self.lqr_start.ricatti()
                self.lqr_end.ricatti()

                # Initialize first control input
                self.us_start = [-self.lqr_start._K[0].dot(pose_0) + self.lqr_start._Kv[0].dot(self.lqr_start._v[0])]
                self.us_end = [-self.lqr_end._K[0].dot(pose_0) + self.lqr_end._Kv[0].dot(self.lqr_end._v[0])]

                # Split model for placing
                _mod1_placing = self.placing_model.gmm_.marginal_array(self.dim1).lintrans(An, bn)
                _mod2_placing = self.placing_model.gmm_.marginal_array(self.dim2).lintrans(Ap, bp)

                # Get the most probable trajectory and uncertainty
                mu1_placing, sigma1_placing = _mod1_placing.condition(self.place_time_axis[:, None], dim_in=slice(0, 1), dim_out=slice(0, 7))
                mu2_placing, sigma2_placing = _mod2_placing.condition(self.place_time_axis[:, None], dim_in=slice(0, 1), dim_out=slice(0, 7))

                # Set up LQR
                self.lqr_start_placing = pbd.LQR(A, b, dt=1.0 / self.publish_rate, horizon=len(mu1_placing))
                self.lqr_start_placing.gmm_xi = [mu1_placing[:, 1:], sigma1_placing[:, 1:, 1:], self.place_sq]
                self.lqr_start_placing.gmm_u = self.reg_factor

                self.lqr_end_placing = pbd.LQR(A, b, dt=1.0 / self.publish_rate, horizon=len(mu2_placing))
                self.lqr_end_placing.gmm_xi = [mu2_placing[:, 1:], sigma2_placing[:, 1:, 1:], self.place_sq]
                self.lqr_end_placing.gmm_u = self.reg_factor

                # Fit LQR
                self.lqr_start_placing.ricatti()
                self.lqr_end_placing.ricatti()

            else:
                print("Invalid controller type")
                return False
            
            # Get current time
            self.start_time = time.time()
            self.duration = float(self.lqr_end.horizon) / self.publish_rate
            self.end_time = self.start_time + self.duration

            # Set up controls for feedback and analysis
            self.controls = [
                np.hstack((self.start_time, self.start_time, self.xis[-1], self.us[-1], self.xi_g[-1]))
            ]

            for time_step_value in self.time_axis:
                self.timer_callback(time_step_value)
            
            for time_step_value in self.place_time_axis:
                self.timer_callback_placing(time_step_value)

            if not self.write_df:
                if self.controller_type == "DualLQR":
                    # Combine two models
                    lqr_list = [self.lqr_start, self.lqr_end, self.lqr_start_placing, self.lqr_end_placing]
                    pickle.dump(
                        lqr_list,
                        open(
                            self.data_folder + "LQR_dual_startTime-{0}.pickle".format(self.start_time),
                            "wb",
                        ),
                    )

                # Combined trajectory elements into array
                all_data = [np.asarray(self.controls, dtype=np.float64), np.asarray(self.transforms), [tf_relaxed_frame_to_ee, tf_relaxed_frame_to_apple, tf_moving_frame_to_place]]

                # Save data
                pickle.dump(
                    all_data,
                    open(
                        self.data_folder + "result_regFactor-{0}_controllerType-{1}_controllerRate-{2}_fruitType-{3}_numberDemonstrations-{4}_startTime-{5}.pickle".format(
                            self.reg_factor,
                            self.controller_type,
                            self.publish_rate,
                            self.fruit_type,
                            self.num_demos,
                            self.start_time,
                        ),
                        "wb",
                    ),
                )
            else:
                row = {}
                row["fruit_type"] = self.fruit_type
                row["number_demonstrations"] = self.num_demos
                row["action_type"] = "reproduction"
                row["time_stamp"] = self.start_time

                # Get control data of run
                control_data = np.asarray(self.controls, dtype=np.float64)

                # start_time_loop = control_data[:,0]
                # end_time_loop = control_data[:,1]
                xis = control_data[:,2:8]
                # us = control_data[:,8:14]
                xi_g = control_data[:,14:21]

                # Determine fruit pose
                fruit_quat = t3d.quaternions.mat2quat(tf_relaxed_frame_to_apple[0:3, 0:3])
                row["fruit_pos_x"] = tf_relaxed_frame_to_apple[0, 3]
                row["fruit_pos_y"] = tf_relaxed_frame_to_apple[1, 3]
                row["fruit_pos_z"] = tf_relaxed_frame_to_apple[2, 3]
                row["fruit_quat_w"] = fruit_quat[0]
                row["fruit_quat_x"] = fruit_quat[1]
                row["fruit_quat_y"] = fruit_quat[2]
                row["fruit_quat_z"] = fruit_quat[3]

                # Placement pose is hardcoded in reproductions
                row["placement_pos_x"] = -0.075
                row["placement_pos_y"] = 0.425
                row["placement_pos_z"] = 0.255
                row["placement_quat_w"] = 0.411
                row["placement_quat_x"] = -0.912
                row["placement_quat_y"] = 0.0
                row["placement_quat_z"] = 0.0

                xi_g_min_section_time_indices = np.argsort(xi_g[:,6])[0:5]
                # Get lowest index that is more than half the horizon of the approach model
                placing_start_index = xi_g_min_section_time_indices[np.where(xi_g_min_section_time_indices>(self.lqr_end.horizon/2.0))[0][0]]

                row["approach_time"] = (placing_start_index-1)/self.publish_rate
                row["total_time"] = len(xis)/self.publish_rate
                row["placing_time"] = (len(xis) - (placing_start_index-1))/self.publish_rate

                # Get approach and picking segment
                if row["fruit_type"] == "apple" and row["number_demonstrations"] == 5:
                    data_range = []
                    for pose_element in xis:
                        quat = t3d.euler.euler2quat(pose_element[3], pose_element[4], pose_element[5])
                        ori = np.asarray(t3d.euler.quat2euler(quat, axes="syzx"))
                        roll, pitch, yaw = ori[[2, 1, 0]]
                        data_range.append(np.hstack((pose_element[:3], [roll, pitch, yaw])))
                    data_range = np.asarray(data_range)
                    data_picking = data_range[placing_start_index:]
                    data_approach = data_range[:placing_start_index]
                else:
                    data_picking = xis[placing_start_index:]
                    data_approach = xis[:placing_start_index]
                
                #### Detachment motion calculation
                # Set fruit pose
                fruit_ori = np.asarray(t3d.euler.mat2euler(tf_relaxed_frame_to_apple[0:3, 0:3], axes="syzx"))
                fruit_roll, fruit_pitch, fruit_yaw = fruit_ori[[2, 1, 0]]
                fruit_pose = np.array([tf_relaxed_frame_to_apple[0, 3], tf_relaxed_frame_to_apple[1, 3], tf_relaxed_frame_to_apple[2, 3], fruit_roll, fruit_pitch, fruit_yaw])
                data_picking_fruit = np.apply_along_axis(pbd.utils.transform_matrix_3D, 1, data_picking, fruit_pose)

                # Calculate the max orientation change
                if row["fruit_type"] == "apple":
                    max_ori_change = np.max(data_picking_fruit[:,3], axis=0)
                elif row["fruit_type"] == "pear":
                    max_ori_change = np.min(data_picking_fruit[:,4], axis=0)
                row["max_ori_change"] = max_ori_change

                ### Final approach accuracy calculation
                data_approach_end = np.apply_along_axis(pbd.utils.transform_matrix_3D, 1, data_approach, fruit_pose)
                try: 
                    # Trim each trajectory to the section where y is greater than -0.075 until the end
                    # Find the index where y is greater than -0.075
                    idx = np.where((data_approach_end[:,1] > -0.075) & (data_approach_end[:,1] < 0.075))[0]
                    # Find the index whereafter y is less than -0.075 continuously
                    idx_diffs = np.where(np.diff(idx) > 1)[0]
                    if len(idx_diffs) > 0:
                        idx = idx[idx_diffs[-1]+1]
                    else:
                        idx = idx[0]
                    # Append the trimmed trajectory
                    data_approach_end_trimmed = data_approach_end[idx:,:]

                    # Get mean offset in each direction
                    mae = np.mean(np.abs(data_approach_end_trimmed), axis=0)
                    # Add to dataframe
                    row["mae_x_0.075"] = mae[0]
                    row["mae_y_0.075"] = mae[1]
                    row["mae_z_0.075"] = mae[2]
                    row["mae_roll_0.075"] = mae[3]
                    row["mae_pitch_0.075"] = mae[4]
                    row["mae_yaw_0.075"] = mae[5]
                    row["num_close_points_0.075"] = len(data_approach_end_trimmed)
                except:
                    row["mae_x_0.075"] = np.nan
                    row["mae_y_0.075"] = np.nan
                    row["mae_z_0.075"] = np.nan
                    row["mae_roll_0.075"] = np.nan
                    row["mae_pitch_0.075"] = np.nan
                    row["mae_yaw_0.075"] = np.nan
                    row["num_close_points_0.075"] = 0

                ### Final pose offset
                final_offset = data_approach_end[-1]
                # Add to dataframe    
                row["final_offset_x"] = final_offset[0]
                row["final_offset_y"] = final_offset[1]
                row["final_offset_z"] = final_offset[2]
                row["final_offset_roll"] = final_offset[3]
                row["final_offset_pitch"] = final_offset[4]
                row["final_offset_yaw"] = final_offset[5]

                for final_fraction in self.final_fractions:
                    try:
                        # Trim each trajectory to the section based on final fraction
                        idx = int(final_fraction * data_approach_end.shape[0])
                        # Append the trimmed trajectory
                        data_approach_end_trimmed = data_approach_end[-idx:,:]

                        # Get mean offset in each direction
                        mae = np.mean(np.abs(data_approach_end_trimmed), axis=0)
                        # Add to dataframe
                        row["mae_x_fraction_"+str(final_fraction)] = mae[0]
                        row["mae_y_fraction_"+str(final_fraction)] = mae[1]
                        row["mae_z_fraction_"+str(final_fraction)] = mae[2]
                        row["mae_roll_fraction_"+str(final_fraction)] = mae[3]
                        row["mae_pitch_fraction_"+str(final_fraction)] = mae[4]
                        row["mae_yaw_fraction_"+str(final_fraction)] = mae[5]
                        row["num_close_points_fraction_"+str(final_fraction)] = len(data_approach_end_trimmed)
                    except:
                        row["mae_x_fraction_"+str(final_fraction)] = np.nan
                        row["mae_y_fraction_"+str(final_fraction)] = np.nan
                        row["mae_z_fraction_"+str(final_fraction)] = np.nan
                        row["mae_roll_fraction_"+str(final_fraction)] = np.nan
                        row["mae_pitch_fraction_"+str(final_fraction)] = np.nan
                        row["mae_yaw_fraction_"+str(final_fraction)] = np.nan
                        row["num_close_points_fraction_"+str(final_fraction)] = 0

                # Write row to csv file
                df = pd.DataFrame([row])
                df.to_csv(self.data_folder + self.df_file_name, mode='a', header=False, index=False)

            return success
        
        except Exception as e:
            print("Error in control initialization: {0}".format(e))
            return False

if __name__ == '__main__':
    SimPickExecutor(sys.argv[1:])