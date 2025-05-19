#!/usr/bin/env python3

# This file listens for the command to start executing a trajectory
# It will publish goal poses at a define rate

# Inputs: JointStates from UR, target object pose

# Author: Robert van de Ven
# Email: robert.vandeven@wur.nl

# Import modules
import numpy as np
import pickle
import transforms3d as t3d
import sys
import time
import roboticstoolbox as rtb
import spatialmath as sm
import matplotlib.pyplot as plt
from copy import deepcopy

import yaml
import os

# Locally import UR5e model
from rtb_model.UR5e import UR5e

# UR rtde imports
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
import psutil

# Imports for natnetclient
import time
from natnet_client import DataDescriptions, DataFrame, NatNetClient

class TrajectoryExecutorReactive(object):
    def __init__(self, sysArgs):
        # Helper variables
        self.publish_rate = 120.0
        self.data_folder = "/media/data/DataRobert/PhD/05 Study 4/04 Robot/teleop_trajectories/"
        self.rigid_bodies = None
        self.tf_world_to_glove = None
        self.tf_base_link_to_world = None
        self.tf_base_link_to_glove = None
        self.tf_glove_to_tcp = None
        self.config_file = "abe_ur5e.yaml"
        self.transforms = []
        self.attached_suction = True

        # Parameters
        self.acc = 1.5 # Has no actual influence on the robot
        self.vel = 3.0 # Has no actual influence on the robot
        self.lookahead_time = 0.03 # Tune this parameter
        self.gain = 200 # Tune this parameter
        self.rtde_frequency = self.publish_rate
        self.dt = 1.0/self.publish_rate 
        self.flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_USE_EXT_UR_CAP
        self.ur_cap_port = 50002
        self.robot_ip = "192.168.10.5"

        # Set up learned frames
        self.get_relaxed_frame()
        self.get_tcp_frame()

        # Load UR5e model for IK solver
        self.ur5e = UR5e()
        self.ur5e_ets = self.ur5e.ets()

        # Set up UR rtde
        self.set_up_rtde()
        
        # Start up NatNetClient
        self.start_natnet()

        # Reset the robot to the home position for this task
        self.reset_manipulator()

        while self.tf_world_to_glove is None:
            time.sleep(0.1)
        self.calib_opti_glove()
        while self.tf_base_link_to_glove is None:
            time.sleep(0.1)

        print("Starting up TeleOp Executor")
        self.intialize_control()

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

    def get_relaxed_frame(self):
        # Set transformation matrix
        tx = -0.240
        ty = 0.167
        tz = 0.565
        rx = 0.0
        ry = 0.0
        rz = -0.707
        rw = 0.707

        tf_base_link_to_relaxed = np.eye(4)
        tf_base_link_to_relaxed[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
        tf_base_link_to_relaxed[0:3, 3] = np.asarray((tx, ty, tz))

        self.tf_base_link_to_relaxed = tf_base_link_to_relaxed
    
    def get_tcp_frame(self):
        # Set transformation matrix
        tx = 0.0
        ty = 0.0
        tz = 0.0
        rx = 0.0
        ry = 0.707
        rz = -0.707
        rw = 0.0

        tf_tcp_to_ee = np.eye(4)
        tf_tcp_to_ee[0:3, 0:3] = t3d.quaternions.quat2mat((rw, rx, ry, rz))
        tf_tcp_to_ee[0:3, 3] = np.asarray((tx, ty, tz))

        self.tf_tcp_to_ee = tf_tcp_to_ee

    def calib_opti_glove(self):
        # Use current glove pose and manipulator pose to determine transformation
        # Get current manipulator pose
        joint_positions = self.rtde_r.getActualQ()
        tf_base_to_tcp = np.asarray(self.ur5e_ets.fkine(joint_positions))
        tf_world_to_glove = deepcopy(self.tf_world_to_glove)

        # Correct translation between robot and optitrack
        translation_base_link_to_world = tf_base_to_tcp[0:3, 3] - tf_world_to_glove[0:3, 3]
        tf_base_link_to_world = np.eye(4)
        tf_base_link_to_world[0:3, 3] = translation_base_link_to_world
        self.tf_base_link_to_world = tf_base_link_to_world  

        # Correct orientation between glove and tcp
        rotation_glove_to_tcp = np.linalg.multi_dot([np.linalg.inv(tf_base_to_tcp[0:3, 0:3]), tf_world_to_glove[0:3, 0:3]])
        tf_glove_to_tcp = np.eye(4)
        tf_glove_to_tcp[0:3, 0:3] = rotation_glove_to_tcp
        self.tf_glove_to_tcp = np.linalg.inv(tf_glove_to_tcp)
        # print("TF glove to tcp: ", self.tf_glove_to_tcp)

        # np.linalg.multi_dot([tf_base_to_tcp, np.linalg.inv(self.tf_world_to_glove)])

    def receive_new_frame(self, data_frame: DataFrame):
        # Get rigid bodies
        self.rigid_bodies = data_frame.rigid_bodies

        for rigid_body in data_frame.rigid_bodies:
            if rigid_body.id_num == 1009 and rigid_body.tracking_valid: #TODO: Determine glove tracker ID
                # Get position and rotation
                pos = rigid_body.pos
                rot = rigid_body.rot

                # Get transformation matrix
                rot_matrix = t3d.quaternions.quat2mat((rot[3], rot[0], rot[1], rot[2]))

                # Get transformation matrix
                matrix = np.eye(4)
                matrix[0:3, 0:3] = rot_matrix
                matrix[0:3, 3] = pos

                self.tf_world_to_glove = matrix
                if rigid_body.marker_error > 0.001: print("Glove error: ", rigid_body.marker_error)
        
        if self.tf_world_to_glove is not None and self.tf_base_link_to_world is not None and self.tf_glove_to_tcp is not None:
            # Calculate transformation matrix between base link and cube
            self.tf_base_link_to_glove = np.linalg.multi_dot([self.tf_base_link_to_world, self.tf_world_to_glove, self.tf_glove_to_tcp])

        # Start up logging
        self.transform_logger()
        
    def receive_new_desc(self, desc: DataDescriptions):
        print("Received data descriptions.")

    def start_natnet(self):
        self.streaming_client = NatNetClient(server_ip_address="192.168.10.1", local_ip_address="192.168.10.70", use_multicast=True)
        self.streaming_client.on_data_description_received_event.handlers.append(self.receive_new_desc)
        self.streaming_client.on_data_frame_received_event.handlers.append(self.receive_new_frame)

        self.streaming_client.connect()
        self.streaming_client.request_modeldef()
        self.streaming_client.run_async()

    def reset_manipulator(self):
        # Determine target tcp, which is relaxed frame for ee
        target_tcp = np.linalg.multi_dot([self.tf_base_link_to_relaxed, np.linalg.inv(self.tf_tcp_to_ee)])
        
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

        self.rtde_c.moveJ(joint_positions)

        return True

    def transform_logger(self):
        # Log transform between base and dynamic frame
        try:
            tf_relaxed_frame_to_ee = np.linalg.multi_dot([np.linalg.inv(self.tf_base_link_to_relaxed), np.asarray(self.ur5e_ets.fkine(self.rtde_r.getActualQ())), self.tf_tcp_to_ee])
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
        # Get current state
        current_time = time.time()
        
        # Check if dead man switch is pressed
        if self.rtde_r.getDigitalInState(1) and not self.dead_man_has_been_pressed:
            # print("DM switch not pressed, waiting before execution")
            return True
        elif self.rtde_r.getDigitalInState(1) and self.dead_man_has_been_pressed:
            print("DM switch released, stopping execution")
            return False
        elif not self.rtde_r.getDigitalInState(1) and not self.dead_man_has_been_pressed:
            print("DM switch pressed, starting execution")
            self.calib_opti_glove()
            self.dead_man_has_been_pressed = True

            # Turn on vacuum pump and open suction
            self.rtde_io.setStandardDigitalOut(2, True) # Turn on vacuum pump
            self.rtde_io.setStandardDigitalOut(1, True) # Open suction

            return True
        else: 
            # Execution case
            # Use TracIK to solve IK
            joint_positions, search_success, iterations, searches, residual = self.ur5e_ets.ik_LM(
                self.tf_base_link_to_glove,
                q0=self.rtde_r.getActualQ(),
                ilimit=30,  # Maximum iterations allowed in a search for a solution
                slimit=100,  # Maximum searches allowed for a solution
                tol=1e-6,  # Solution tolerance
                k=0.1,
                joint_limits=False,
                method="chan",
            )

            pre_chamber_pressure = self.rtde_r.getStandardAnalogInput0()
            post_chamber_pressure = self.rtde_r.getStandardAnalogInput1()

            # Move robot to new position
            self.rtde_c.servoJ(joint_positions, self.vel, self.acc, self.dt, self.lookahead_time, self.gain)

            # Store controls
            self.controls += [
                np.hstack((current_time, time.time(), joint_positions, search_success, iterations, searches, residual, pre_chamber_pressure, post_chamber_pressure))
            ]

            if not self.rtde_r.getDigitalInState(0):
                # Close suction and open release valve
                self.rtde_io.setStandardDigitalOut(1, False) # Close suction
                self.rtde_io.setStandardDigitalOut(0, True) # Open release valve

            return True
    
    def intialize_control(self):
        # Set success boolean
        success = True
        current_time = time.time()
        self.dead_man_has_been_pressed = False

        # Set up controls for feedback and analysis
        self.controls = [
            np.hstack((current_time, time.time(), self.rtde_r.getActualQ(), 1, 0, 1, 0.0, 0.0, 0.0))
        ]

        try:
            while success:
                t_start = self.rtde_c.initPeriod()
                success = self.timer_callback()
                self.rtde_c.waitPeriod(t_start)
        except KeyboardInterrupt:
            print("Control Interrupted!")

        print("Shutting down TeleOp Executor")
        
        # Combined trajectory elements into array
        all_data = [np.asarray(self.controls, dtype=np.float64), np.asarray(self.transforms)]

        self.streaming_client.shutdown()
        # Turn off vacuum pump and other outputs
        self.rtde_io.setStandardDigitalOut(2, False)
        self.rtde_io.setStandardDigitalOut(1, False)
        self.rtde_io.setStandardDigitalOut(0, False)

        # Key press to indicate success
        print("Press 'n' to save as failed, 'y' to save as success")
        key = input()
        if key == "n" or key == "N":
            print("Failed demonstration")
            success_value = "False"
        elif key == "y" or key == "Y" or key == "": 
            print("Successful demonstration")
            success_value = "True"
        else:
            print("Unknown input, saving as None")
            success_value = "None"
        
        # Save data
        pickle.dump(
            all_data,
            open(
                self.data_folder + "teleop_successValue-{0}_controllerRate-{1}_lookaheadTime-{2}_gain-{3}_startTime-{4}.pickle".format(
                    success_value,
                    self.publish_rate,
                    self.lookahead_time,
                    self.gain,
                    current_time,
                ),
                "wb",
            ),
        )

        return success

if __name__ == "__main__":
    TrajectoryExecutorReactive(sys.argv[1:])