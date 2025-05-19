import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PointStamped

# Import modules
import numpy as np
import pickle
import transforms3d as t3d
import pbdlib as pbd
import sys
import time
import roboticstoolbox as rtb
from scipy.linalg import solve_discrete_are as solve_algebraic_riccati_discrete
from threading import Thread
from copy import deepcopy
import signal

import os

# Locally import UR5e model
from .rtb_model.UR5e import UR5e

# UR rtde imports
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
import psutil

# Imports for natnetclient
import time

# Rel import
from .pbdlib_custom import gmmr

# Fix the change in structure between pickle and ROS2 structure
sys.modules["gmmr"] = gmmr

class ApplePickExecutor(Node):

    def __init__(self):
        super().__init__('apple_pick_executor')

        # Create a callback group to prevent the subscription's callback from being blocked by the service's callback
        self._default_callback_group = ReentrantCallbackGroup()

        # # Create subscriber for apple pose
        # self.tf_base_link_to_apple = None
        # self.update_apple_pose = False
        # self.apple_pose_sub = self.create_subscription(
        #     PointStamped,
        #     '/apple/marker_front',
        #     self.apple_pose_callback,
        #     10)
        # self.apple_pose_sub  # prevent unused variable warning

        # Helper variables
        self.publish_rate = 120.0
        self.controller_rate = 120.0
        self.controller_type = "DualLQR"
        self.data_folder = "/media/data/DataRobert/PhD/05 Study 4/04 Robot/orchard_apple_trajectories/"
        self.model_name = "/media/data/DataRobert/PhD/05 Study 4/04 Robot/test_setup/apple_20_demos/model/approach/gmr_demos-20_LL-10.30108528510114_depMask-Split_trainTime-1725457292.3484292.pickle"
        self.placing_model_name = "/media/data/DataRobert/PhD/05 Study 4/04 Robot/test_setup/apple_20_demos/model/placing/gmr_demos-20_LL-3.0090913478028294_depMask-Split_trainTime-1725457360.4215798.pickle"
        self.rigid_bodies = None
        self.config_file = "abe_ur5e.yaml"
        self.reg_factor = -1.5
        self.transforms = []
        self.attached_suction = False
        self.rot_convention = "syzx"
        self.column_reorder = [2, 1, 0]

        # Parameters
        self.acc = 1.5 # Determine what is desired
        self.vel = 3.0 # Determine what is desired
        self.rtde_frequency = self.publish_rate
        self.dt = 1.0/self.publish_rate 
        self.flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_USE_EXT_UR_CAP
        self.ur_cap_port = 50002
        self.robot_ip = "192.168.10.5"

        self.lookahead_time = 0.03
        self.gain = 400

        # Set up frames
        self.get_camera_frame()
        self.get_relaxed_frame()
        # self.get_tcp_frame()
        # self.get_apple_grasp_frame()
        self.get_start_frame()
        self.get_base_link_frame()
        self.get_place_frame()

        # Load UR5e model for IK solver
        self.ur5e = UR5e()
        # self.ur5e.qlim = np.array([[ 0.00000000, -6.28318531, -3.14159265, -6.28318531, -6.28318531, -6.28318531],
        #                            [ 6.28318531,  6.28318531,  3.14159265,  6.28318531,  6.28318531,  6.28318531]])
        self.ur5e_ets = self.ur5e.ets()
        # print(self.ur5e_ets.qlim)

        # Set up UR rtde
        self.set_up_rtde()

        print("Starting up Trajectory Executor")
        self.control_running = True
        self.thread_control = Thread(target=self.intialize_control)
        self.thread_control.start()

    # def apple_pose_callback(self, msg):
    #     if self.update_apple_pose:
    #         if not np.isnan(msg.point.x):
    #             tf_camera_to_apple = np.eye(4)
    #             tf_camera_to_apple[0:3, 3] = np.asarray((msg.point.x, msg.point.y, msg.point.z))

    #             tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(self.rtde_r.getActualQ()))
    #             tf_base_link_to_apple = np.linalg.multi_dot([tf_base_to_tcp, self.tf_tcp_to_camera, tf_camera_to_apple])
           
    #             self.tf_base_link_to_apple = tf_base_link_to_apple

    def set_up_rtde(self):
        # ur_rtde realtime priorities
        rt_receive_priority = 90
        rt_control_priority = 85

        setup = False
        while not setup:
            try:
                self.rtde_r = RTDEReceive(self.robot_ip, self.rtde_frequency, [], True, False, rt_receive_priority)
                self.rtde_c = RTDEControl(self.robot_ip, self.rtde_frequency, self.flags, self.ur_cap_port, rt_control_priority)
                self.rtde_io = RTDEIO(self.robot_ip)
                setup = True
            except:
                print("Failed to setup RTDE, retrying in 1 second")
                time.sleep(1)

        # Set application real-time priority
        os_used = sys.platform
        process = psutil.Process(os.getpid())
        if os_used == "win32":  # Windows (either 32-bit or 64-bit)
            process.nice(psutil.REALTIME_PRIORITY_CLASS)
        elif os_used == "linux":  # linux
            rt_app_priority = 80
            param = os.sched_param(rt_app_priority)
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, param)
            except OSError:
                print("Failed to set real-time process scheduler to %u, priority %u" % (os.SCHED_FIFO, rt_app_priority))
            else:
                print("Process real-time priority set to: %u" % rt_app_priority)

    def get_camera_frame(self):
        tf_tcp_to_camera_link = np.asarray([[-0.015, -0.999, -0.028, -0.006],
                                            [ 0.010,  0.028, -1.000, -0.112],
                                            [ 1.000, -0.015,  0.010,  0.091],
                                            [ 0.000,  0.000,  0.000,  1.000]])
        # tf_tcp_to_camera_link[0:3, 3] = np.asarray((-0.006, -0.108, 0.094))
        # qx = 0.48898
        # qy = -0.513841
        # qz = 0.491003
        # qw = 0.505749
        # tf_tcp_to_camera_link[0:3, 0:3] = t3d.quaternions.quat2mat((qw, qx, qy, qz))

        tf_camera_link_to_center = np.eye(4)
        tf_camera_link_to_center[0:3, 3] = np.asarray((0.0, 0.0, 0.016))

        tf_camera_center_to_left = np.eye(4)
        tf_camera_center_to_left[0:3, 3] = np.asarray((-0.01, 0.025, 0.0))

        tf_camera_left_to_optical = np.eye(4)
        tf_camera_left_to_optical[0:3, 0:3] = t3d.quaternions.quat2mat((0.5, -0.5, 0.5, -0.5))

        self.tf_tcp_to_camera = np.linalg.multi_dot([tf_tcp_to_camera_link, tf_camera_link_to_center, tf_camera_center_to_left, tf_camera_left_to_optical])
    
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
        # Set transformation matrix
        tx = 0.0
        ty = -0.2
        tz = 0.0
        rx = 0.0
        ry = 0.0
        rz = -0.707
        rw = 0.707

        tf_base_link_to_relaxed = np.eye(4)
        tf_base_link_to_relaxed[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
        tf_base_link_to_relaxed[0:3, 3] = np.asarray((tx, ty, tz))

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
    
    # def get_tcp_frame(self):
    #     # Set transformation matrix
    #     tx = 0.0
    #     ty = 0.0
    #     tz = 0.0
    #     rx = 0.0
    #     ry = 0.707
    #     rz = -0.707
    #     rw = 0.0

    #     tf_tcp_to_ee = np.eye(4)
    #     tf_tcp_to_ee[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
    #     tf_tcp_to_ee[0:3, 3] = np.asarray((tx, ty, tz))

    #     self.tf_tcp_to_ee = tf_tcp_to_ee

    # def get_apple_grasp_frame(self):
    #     # Set transformation matrix
    #     tx = 0.0
    #     ty = 0.0
    #     tz = 0.0
    #     rx = 0.0
    #     ry = 0.707
    #     rz = -0.707
    #     rw = 0.0

    #     tf_apple_to_ee = np.eye(4)
    #     tf_apple_to_ee[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
    #     tf_apple_to_ee[0:3, 3] = np.asarray((tx, ty, tz))

    #     self.tf_apple_to_ee = tf_apple_to_ee

    def get_apple_from_manipulator(self):
        input("Put the robot at the grasp pose and press enter")
        # Get initial manipulator pose
        joint_positions = self.rtde_r.getActualQ()
        # Calculate forward kinematics
        tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(joint_positions))
        # TF matrix going slightly forward
        tf_tcp_to_apple = np.eye(4)
        tf_tcp_to_apple[0:3, 3] = np.asarray((0.0, 0.0, 0.02))

        self.transform_logger()

        # Set as tf between base and apple
        self.tf_base_link_to_apple = np.linalg.multi_dot([tf_base_to_tcp, tf_tcp_to_apple])

        input("Put the robot close to the relaxed pose and press enter")

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
        self.tf_base_link_to_place = np.dot(np.asarray([[ 0.,   1.,   0.,   0.],
                                                        [-1.,   0.,  -0.,   0. ],
                                                        [-0.,   0.,   1.,   0. ],
                                                        [ 0.,   0.,   0.,   1. ],]), tf_world_to_place)

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

    def reset_manipulator(self):
        # Determine target tcp, which is relaxed frame for ee
        target_tcp = np.linalg.multi_dot([self.tf_base_link_to_relaxed, self.tf_relaxed_to_init_manip])

        # Calculate IK
        joint_positions, search_success, iterations, searches, residual = self.ur5e_ets.ik_LM(
            target_tcp,
            q0=self.rtde_r.getActualQ(),
            ilimit=30,  # Maximum iterations allowed in a search for a solution
            slimit=100,  # Maximum searches allowed for a solution
            tol=1e-6,  # Solution tolerance
            k=0.1,
            joint_limits=False,
            method="chan",
        )

        self.rtde_c.moveJ(joint_positions, speed=0.5, acceleration=0.5)

        return True
    
    # def perception_pose(self):
    #     # Define target tcp
    #     target_tcp = np.eye(4)
    #     target_tcp[0:3, 3] = np.asarray((-0.447, 0.066, 0.223))
    #     target_tcp[0:3, 0:3] = t3d.quaternions.quat2mat((0.5, -0.5, -0.5, 0.5))
        
    #     # Calculate IK
    #     joint_positions, search_success, iterations, searches, residual = self.ur5e_ets.ik_LM(
    #         target_tcp,
    #         q0=self.rtde_r.getActualQ(),
    #         ilimit=30,  # Maximum iterations allowed in a search for a solution
    #         slimit=100,  # Maximum searches allowed for a solution
    #         tol=1e-6,  # Solution tolerance
    #         k=0.1,
    #         joint_limits=False,
    #         method="chan",
    #     )

    #     self.rtde_c.moveJ(joint_positions, speed=0.5, acceleration=0.5)            

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

    def timer_callback(self):
        # Time since started
        current_time = time.time()

        diff = current_time - self.start_time

        # Fraction of time
        time_fract = diff / self.duration

        # Integer time step
        t = int(time_fract * (self.lqr_end.horizon - 2))

        if t > self.lqr_end.horizon - 2:
            print("Time step too large")  
            t = self.lqr_end.horizon - 2

        # Get current state
        # Calculate forward kinematics
        tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(self.rtde_r.getActualQ()))
        # Calculate matrix between ee and moving frame
        tf_moving_frame_to_ee = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), tf_base_to_tcp])
        # Calculate pose
        pose_i = self.matrix_to_pose(tf_moving_frame_to_ee, axes=self.rot_convention, reorder=self.column_reorder)
        self.xis += [pose_i] # [self.xi_g[-1][:-1]] # 

        if self.controller_type == "SingleLQR":
            # Control input based on current pose
            self.us_end += [-self.lqr_end._K[t].dot(self.xis[-1]) + self.lqr_end._Kv[t].dot(self.lqr_end._v[t + 1])]
            self.us += [self.us_end[-1]]

        elif self.controller_type == "DualLQR":
            # Get current state in start frame
            tf_ee_to_relaxed_frame = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), tf_base_to_tcp])
            pose_i_start = self.matrix_to_pose(tf_ee_to_relaxed_frame, axes=self.rot_convention, reorder=self.column_reorder)

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
                time_fract,
            )
        ]  

        # Determine goal pose from robot base frame to tcp
        goal_moving_frame_to_ee = self.array_to_matrix(
            self.xi_g[-1][:-1],
            axes=self.rot_convention,
            reorder=self.column_reorder,
        )
        # Calculate goal pose in robot base frame
        goal_base_to_tcp = np.linalg.multi_dot([self.tf_base_link_to_relaxed, goal_moving_frame_to_ee])

        # Use TracIK to solve IK
        joint_positions, search_success, iterations, searches, residual = self.ur5e_ets.ik_LM(
            goal_base_to_tcp,
            q0=self.rtde_r.getActualQ(),
            ilimit=30,  # Maximum iterations allowed in a search for a solution
            slimit=100,  # Maximum searches allowed for a solution
            tol=1e-6,  # Solution tolerance
            k=0.1,
            joint_limits=True,
            method="chan",
        )

        pre_chamber_pressure = self.rtde_r.getStandardAnalogInput0()
        post_chamber_pressure = self.rtde_r.getStandardAnalogInput1()

        if post_chamber_pressure < 2.0:
            self.rtde_c.servoJ(joint_positions, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)
        else:
            # Get current state 
            tf_relaxed_to_apple = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), self.tf_base_link_to_apple])
            tf_base_to_grasp_ee = deepcopy(tf_base_to_tcp)
            self.tf_base_link_to_relaxed_grasp = np.linalg.multi_dot([tf_base_to_grasp_ee, np.linalg.inv(tf_relaxed_to_apple)])
            self.end_time = time.time() - 0.2


        # Determine goal in relaxed frame
        goal_pose = self.matrix_to_pose(goal_moving_frame_to_ee, axes=self.rot_convention, reorder=self.column_reorder)

        # Store controls
        self.controls += [
            np.hstack((current_time, time.time(), self.xis[-1], self.us[-1], self.xi_g[-1], joint_positions, search_success, iterations, searches, residual, goal_pose, pre_chamber_pressure, post_chamber_pressure))
        ]
    
    def timer_callback_placing(self):
        # Time since started
        current_time = time.time()

        diff = current_time - self.start_time_placing

        # Fraction of time
        time_fract = diff / self.duration_placing

        # Integer time step
        t = int(time_fract * (self.lqr_end_placing.horizon - 2))

        if t > self.lqr_end_placing.horizon - 2:
            print("Time step too large")  
            t = self.lqr_end_placing.horizon - 2

        # Get current state
        # Calculate forward kinematics
        tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(self.rtde_r.getActualQ()))
        # Calculate matrix between ee and moving frame when apple was grasped
        tf_moving_frame_to_ee = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), tf_base_to_tcp])
        # Calculate pose
        pose_i = self.matrix_to_pose(tf_moving_frame_to_ee, axes=self.rot_convention, reorder=self.column_reorder)
        self.xis += [pose_i] # [self.xi_g[-1][:-1]] # 

        if self.controller_type == "SingleLQR":
            # Control input based on current pose
            self.us_end += [-self.lqr_end_placing._K[t].dot(self.xis[-1]) + self.lqr_end_placing._Kv[t].dot(self.lqr_end_placing._v[t + 1])]
            self.us += [self.us_end[-1]]

        elif self.controller_type == "DualLQR":
            # Get current state in start frame
            tf_ee_to_relaxed_frame = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed_grasp), tf_base_to_tcp])
            pose_i_start = self.matrix_to_pose(tf_ee_to_relaxed_frame, axes=self.rot_convention, reorder=self.column_reorder)

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
                time_fract,
            )
        ]

        # Determine goal pose from robot base frame to tcp
        goal_moving_frame_to_ee = self.array_to_matrix(
            self.xi_g[-1][:-1],
            axes=self.rot_convention,
            reorder=self.column_reorder,
        )
        # Calculate goal pose in robot base frame
        goal_base_to_tcp = np.linalg.multi_dot([self.tf_base_link_to_relaxed, goal_moving_frame_to_ee])

        # Use TracIK to solve IK
        joint_positions, search_success, iterations, searches, residual = self.ur5e_ets.ik_LM(
            goal_base_to_tcp,
            q0=self.rtde_r.getActualQ(),
            ilimit=30,  # Maximum iterations allowed in a search for a solution
            slimit=100,  # Maximum searches allowed for a solution
            tol=1e-6,  # Solution tolerance
            k=0.1,
            joint_limits=True,
            method="chan",
        )

        pre_chamber_pressure = self.rtde_r.getStandardAnalogInput0()
        post_chamber_pressure = self.rtde_r.getStandardAnalogInput1()

        self.rtde_c.servoJ(joint_positions, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)

        # Determine goal in relaxed frame
        goal_pose = self.matrix_to_pose(goal_moving_frame_to_ee, axes=self.rot_convention, reorder=self.column_reorder)

        # Store controls
        self.controls += [
            np.hstack((current_time, time.time(), self.xis[-1], self.us[-1], self.xi_g[-1], joint_positions, search_success, iterations, searches, residual, goal_pose, pre_chamber_pressure, post_chamber_pressure))
        ]

    def intialize_control(self):
        try:
            # Set success boolean
            success = True

            # Get apple pose
            self.get_apple_from_manipulator()

            # Set the robot to perception pose
            # self.perception_pose()

            # self.update_apple_pose = True

            # # Wait for apple pose
            # while self.tf_base_link_to_apple is None and self.control_running:
            #     print("Waiting for apple pose")
            #     time.sleep(0.1)

            # self.update_apple_pose = False

            # print("Base apple: \n {} ".format(self.tf_base_link_to_apple))

            # Reset the robot to the home position for this task
            self.reset_manipulator()

            # Get tcp
            joint_positions = self.rtde_r.getActualQ()
            # Calculate forward kinematics
            tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(joint_positions))
            
            # Get tf between ee and relaxed_frame
            tf_relaxed_frame_to_ee = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), tf_base_to_tcp])
            # Calculate pose
            pose_0 = self.matrix_to_pose(tf_relaxed_frame_to_ee, axes=self.rot_convention, reorder=self.column_reorder)
            print("Pose 0: {0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f} {5:0.3f}".format(pose_0[0], pose_0[1], pose_0[2], pose_0[3], pose_0[4], pose_0[5]))

            self.initial_pose = pose_0

            # Get transform between base and apple
            tf_relaxed_frame_to_apple = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), self.tf_base_link_to_apple])
            pose_apple = self.matrix_to_pose(tf_relaxed_frame_to_apple, axes=self.rot_convention, reorder=self.column_reorder)
            print("Pose apple: {0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f} {5:0.3f}".format(pose_apple[0], pose_apple[1], pose_apple[2], pose_apple[3], pose_apple[4], pose_apple[5]))

            # Get tf between place and moving frame, going through grasp_cube
            tf_moving_frame_to_place = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), self.tf_base_link_to_place])
            pose_place = self.matrix_to_pose(tf_moving_frame_to_place, axes=self.rot_convention, reorder=self.column_reorder)
            print("Pose place: {0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f} {5:0.3f}".format(pose_place[0], pose_place[1], pose_place[2], pose_place[3], pose_place[4], pose_place[5]))

            # while self.control_running:
            #     t_start = self.rtde_c.initPeriod()
            #     tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(self.rtde_r.getActualQ()))
            #     tf_base_to_ee = np.linalg.multi_dot([tf_base_to_tcp, self.tf_tcp_to_ee])
            #     tf_base_to_apple_ee = np.linalg.multi_dot([self.tf_base_link_to_apple, self.tf_apple_to_ee])
            #     xyz_offsets = tf_base_to_apple_ee[0:3, 3] - tf_base_to_ee[0:3, 3]
            #     print("EE offsets: \n {0:0.3f} {1:0.3f} {2:0.3f}".format(xyz_offsets[0], xyz_offsets[1], xyz_offsets[2]))
            #     self.rtde_c.waitPeriod(t_start)
            
            # Ask user whether these poses are correct
            user_input = input("Are these poses correct? (Y/n): ")
            if user_input == "n":
                success = False
                return success
            
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
            if self.controller_type == "SingleLQR":
                # Split model
                _mod2 = self.model.gmm_.marginal_array(self.dim2).lintrans(An, bn)

                # Get the most probable trajectory and uncertainty
                mu2, sigma2 = _mod2.condition(self.time_axis[:, None], dim_in=slice(0, 1), dim_out=slice(0, 7))

                # Set up LQR
                self.lqr_end = pbd.LQR(A, b, 1.0 / self.publish_rate, horizon=len(mu2))
                self.lqr_end.gmm_xi = [mu2[:, 1:], sigma2[:, 1:, 1:], self.sq]
                self.lqr_end.gmm_u = self.reg_factor
                self.lqr_end.ricatti()

                # Initialize first control input
                self.us_end = [-self.lqr_end._K[0].dot(pose_0) + self.lqr_end._Kv[0].dot(self.lqr_end._v[0])]

            elif self.controller_type == "DualLQR":
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
                # self.get_logger().info("Fitting LQR")
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
                # self.get_logger().info("Fitting LQR")
                self.lqr_start_placing.ricatti()
                self.lqr_end_placing.ricatti()

            else:
                self.get_logger().error("Invalid controller type")
                return False
            
            # Turn on vacuum pump and open suction
            self.rtde_io.setStandardDigitalOut(2, True) # Turn on vacuum pump
            self.rtde_io.setStandardDigitalOut(1, True) # Open suction

            # Get current time
            self.start_time = time.time()
            self.duration = float(self.lqr_end.horizon) / self.publish_rate
            self.end_time = self.start_time + self.duration

            # Set up controls for feedback and analysis
            self.controls = [
                np.hstack((self.start_time, self.start_time, self.xis[-1], self.us[-1], self.xi_g[-1], joint_positions, 1, 0, 1, 0.0, pose_0, 0.0, 0.0))
            ]

            while time.time() < self.end_time and self.control_running:
                t_start = self.rtde_c.initPeriod()
                self.timer_callback()
                self.rtde_c.waitPeriod(t_start)

            self.start_time_placing = time.time()
            self.duration_placing = float(self.lqr_end_placing.horizon) / self.publish_rate
            self.end_time_placing = self.start_time_placing + self.duration_placing
            
            while time.time() < self.end_time_placing and self.control_running:
                t_start = self.rtde_c.initPeriod()
                self.timer_callback_placing()
                self.rtde_c.waitPeriod(t_start)

            if self.control_running:
                # Close suction and open release valve
                self.rtde_io.setStandardDigitalOut(1, False) # Close suction
                self.rtde_io.setStandardDigitalOut(0, True) # Open release valve

            # Send zero velocity to stop the robot
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
            # self.streaming_client.shutdown()


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
            elif self.controller_type == "SingleLQR":
                # Save model
                pickle.dump(
                    self.lqr_end,
                    open(
                        self.data_folder + "LQR_end_startTime-{0}.pickle".format(self.start_time),
                        "wb",
                    ),
                )

            # Combined trajectory elements into array
            all_data = [np.asarray(self.controls, dtype=np.float64), np.asarray(self.transforms), [tf_relaxed_frame_to_ee, tf_relaxed_frame_to_apple, tf_moving_frame_to_place]]

            # Save data
            pickle.dump(
                all_data,
                open(
                    self.data_folder + "result_regFactor-{0}_controllerType-{1}_controllerRate-{2}_lookaheadTime-{3}_gain-{4}_startTime-{5}.pickle".format(
                        self.reg_factor,
                        self.controller_type,
                        self.publish_rate,
                        self.lookahead_time,
                        self.gain,
                        self.start_time,
                    ),
                    "wb",
                ),
            )

            print("Control finished")

            # Turn of vacuum pump and all valves
            self.rtde_io.setStandardDigitalOut(0, False) # Close release valve
            self.rtde_io.setStandardDigitalOut(1, False) # Close suction
            self.rtde_io.setStandardDigitalOut(2, False) # Turn off vacuum pump
            # Request SIGINT to stop the node
            os.kill(os.getpid(), signal.SIGINT)

            return success
        
        except Exception as e:
            self.get_logger().error("Failed to initialize control: {0}".format(e))
            return False

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()

    apple_pick_executor = ApplePickExecutor()

    executor.add_node(apple_pick_executor)

    try: 
        executor.spin()
    except KeyboardInterrupt:
        apple_pick_executor.control_running = False
        apple_pick_executor.thread_control.join()
        apple_pick_executor.destroy_node()
        executor.shutdown()
    # finally:
    #     executor.shutdown()
        # apple_pick_executor.destroy_node()
        # rclpy.shutdown()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)



if __name__ == '__main__':
    main()