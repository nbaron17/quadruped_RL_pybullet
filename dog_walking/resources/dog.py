import pybullet as p
import os
import math
import numpy as np
import time
from . import motor_model
# from kinematic_model import robotKinematics


class Dog:
    def __init__(self, client, fixed_base=False, start_height=0.2, no_of_actions=12):
        self.client = client
        robotPos = [0, 0, start_height]
        robotScale = 1
        self.dog = p.loadURDF(os.path.abspath(os.path.dirname(__file__)) + '/dog.urdf',
                   basePosition=robotPos,
                   baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=fixed_base,
                   physicsClientId=client)
        self.joint_ids = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.init_joint_angles = np.array(4 * [0.0, 1.0, -1.7])
        # for i, motor_id in enumerate(self.joint_ids):
        #     p.resetJointState(self.dog, motor_id, self.init_joint_angles[i])
        #     p.setJointMotorControl2(bodyIndex=self.dog,
        #                             jointIndex=motor_id,
        #                             controlMode=p.VELOCITY_CONTROL,
        #                             force=0)

            # Create dictionaries for joint and link IDs
        self.jointNameToID = {}
        self.linkNameToID = {}
        self.revoluteID = []
        for j in range(p.getNumJoints(self.dog)):
            info = p.getJointInfo(self.dog, j)
            jointID = info[0]
            jointName = info[1].decode('UTF-8')
            jointType = info[2]
            if (jointType == p.JOINT_REVOLUTE):
                self.jointNameToID[jointName] = info[0]
                self.linkNameToID[info[12].decode('UTF-8')] = info[0]
                self.revoluteID.append(j)


        self.footFR_index = 3
        self.footFL_index = 7
        self.footBR_index = 11
        self.footBL_index = 15

        self.foot_link_ids = [3, 7, 11, 15]  # FR, FL, BR, BL
        self.non_foot_link_ids = [-1, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        # self.robotKinematics = robotKinematics()
        # self.meassure = systemStateEstimator(self.dog) #meassure from simulation

        self.set_foot_friction(foot_friction=1)
        self.set_base_friction(base_friction=1)

        self.joint_limits = [0.785, 1.57, 1.57]
        self.joint_offsets = [0, 0, 0.785]

        self.no_of_actions = no_of_actions

        #robot properties
        """initial foot position"""
        #foot separation (Ydist = 0.16 -> tetta=0) and distance to floor
        Xdist = 0.20
        Ydist = 0.15
        height = 0.15
        #body frame to foot frame vector
        self.bodytoFeet0 = np.matrix([[ Xdist/2 , -Ydist/2 , -height],
                                [ Xdist/2 ,  Ydist/2 , -height],
                                [-Xdist/2 , -Ydist/2 , -height],
                                [-Xdist/2 ,  Ydist/2 , -height]])

        self.T = 0.5 #period of time (in seconds) of every step
        self.offset = np.array([0.5 , 0. , 0. , 0.5]) #defines the offset between each foot step in this order (FR,FL,BR,BL)
        self.bodytoFeet_vecX = 0.
        self.bodytoFeet_vecY = 0.

        self.pos = np.zeros([3])
        self.orn = np.zeros([3])
        # p.setRealTimeSimulation(1)
        p.setTimeStep(0.002)

        self.motor_joints = self.revoluteID
        self.max_linear_velocity = 5
        self.max_angular_velocity = 5
        self.max_joint_vel = 5

        self._kp = 2.0
        self._kd = 0.02
        self._motor_model = motor_model.MotorModel(torque_control_enabled=False,
                                             kp=self._kp,
                                             kd=self._kd)
        self.init_torque_ctrl()
        self.joint_torques = np.zeros(12)

    def init_torque_ctrl(self):
        # Disable the default motor for joints to allow direct torque control
        for i, motor_id in enumerate(self.joint_ids):
            p.setJointMotorControl2(bodyUniqueId=self.dog,
                                    jointIndex=motor_id,
                                    controlMode=p.VELOCITY_CONTROL,
                                    force=0, physicsClientId=self.client)

    def get_ids(self):
        return self.dog, self.client

    def get_link_names_and_ids(self):
        num_joints = p.getNumJoints(self.dog, physicsClientId=self.client)
        link_ids_and_names = []
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.dog, joint_index)
            link_id = joint_info[0]
            link_name = joint_info[12].decode('utf-8')  # Link name is in the 13th field (index 12)
            link_ids_and_names.append((link_id, link_name))
        for link_id, link_name in link_ids_and_names:
            print(f"Link ID: {link_id}, Link Name: {link_name}")

    def set_torque(self, torque):
        for i, motor_id in enumerate(self.joint_ids):
            p.setJointMotorControl2(bodyIndex=self.dog, jointIndex=motor_id,
                                                        controlMode=p.TORQUE_CONTROL,
                                                        force=torque[i],  physicsClientId=self.client)

    def apply_action(self, action):
        # Expects action to be two-dimensional
        # motor_angles = np.array([3.14 * i for i in action]) # Convert normalised action to motor angles

        if self.no_of_actions == 12:
            motor_angles = action
            motor_angles[[0, 3, 6, 9]] *= self.joint_limits[0]
            motor_angles[[1, 4, 7, 10]] *= self.joint_limits[1]
            motor_angles[[2, 5, 8, 11]] *= self.joint_limits[2]
        else:  # no_of_actions == 9
            motor_angles = np.array([0, action[1], action[2], 0, action[3], action[4], 0, action[5], action[6], 0, action[7], action[8]])
            motor_angles[[0, 3, 6, 9]] *= self.joint_limits[0]
            motor_angles[[1, 4, 7, 10]] *= self.joint_limits[1]
            motor_angles[[2, 5, 8, 11]] *= self.joint_limits[2]

        # get positions and velocities
        q = np.array(self.get_joint_angs(normalised=False))
        qdot = np.array(self.get_joint_vels(normalised=False))

        actual_torque, observed_torque = self._motor_model.convert_to_torque(np.array(motor_angles), q, qdot)
        self.set_torque(actual_torque)
        self.joint_torques = actual_torque.copy()
        # print(q[1], qdot[1], actual_torque[1])

        # noise_std = 10 * 3.14 / 180  # standard deviation 10 degrees
        # noise = np.random.normal(0, noise_std, motor_angles.shape)
        motor_angles_noisy = motor_angles # + noise

        # Clip motor_angles to reasonable values
#        motor_angles = max(min(motor_angles, 0.6), -0.6)

        # Set the steering joint positions
#        p.setJointMotorControlArray(self.dog, self.motor_joints,
#                                    controlMode=p.POSITION_CONTROL,
#                                    targetPositions=[steering_angle] * 2,
#                                    physicsClientId=self.client)
#        for i in range(len(self.motor_joints)):
#            p.setJointMotorControl2(self.dog, self.motor_joints[i], p.POSITION_CONTROL, motor_angles[i], force=100)
#
#         FR_angles = [motor_angles[0], motor_angles[1], motor_angles[2]]
#         FL_angles = [motor_angles[3], motor_angles[4], motor_angles[5]]
#         BL_angles = [motor_angles[6], motor_angles[7], motor_angles[8]]
#         BR_angles = [motor_angles[9], motor_angles[10], motor_angles[11]]
#         # FR_angles = [0, motor_angles_noisy[0], motor_angles_noisy[1]]
#         # FL_angles = [0, motor_angles_noisy[2], motor_angles_noisy[3]]
#         # BL_angles = [0, motor_angles_noisy[4], motor_angles_noisy[5]]
#         # BR_angles = [0, motor_angles_noisy[6], motor_angles_noisy[7]]
#         max_force = 5 #N/m
#         for i in range(3):
#             p.setJointMotorControl2(self.dog, i, p.POSITION_CONTROL,
#                                     targetPosition = FR_angles[i] , force = max_force , maxVelocity = self.max_joint_vel, physicsClientId=self.client)
#             p.setJointMotorControl2(self.dog, 4 + i, p.POSITION_CONTROL,
#                                     targetPosition = FL_angles[i] , force = max_force , maxVelocity = self.max_joint_vel, physicsClientId=self.client)
#             p.setJointMotorControl2(self.dog, 8 + i, p.POSITION_CONTROL,
#                                     targetPosition = BR_angles[i] , force = max_force , maxVelocity = self.max_joint_vel, physicsClientId=self.client)
#             p.setJointMotorControl2(self.dog, 12 + i, p.POSITION_CONTROL,
#                                     targetPosition = BL_angles[i] , force = max_force , maxVelocity = self.max_joint_vel, physicsClientId=self.client)

    def rotate_vec(self, vec1, ang):
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        vec2 = np.matmul(R, vec1)
        return vec2

    def norm_base_vels(self):
        lin_vel, ang_vel = p.getBaseVelocity(self.dog, self.client)
        norm_linear_velocity = np.array(lin_vel, dtype=np.float32) / self.max_linear_velocity
        norm_angular_velocity = np.array(ang_vel, dtype=np.float32) / self.max_angular_velocity
        norm_linear_velocity = np.clip(norm_linear_velocity, -1, 1)
        norm_angular_velocity = np.clip(norm_angular_velocity, -1, 1)
        return norm_linear_velocity, norm_angular_velocity

    def get_joint_angs(self, normalised):
        joint_angles = np.array([i[0] for i in p.getJointStates(self.dog, self.joint_ids, physicsClientId=self.client)])
        if normalised:
            joint_angles[[0, 3, 6, 9]] *= 1 / self.joint_limits[0]
            joint_angles[[1, 4, 7, 10]] *= 1 / self.joint_limits[1]
            joint_angles[[2, 5, 8, 11]] *= 1 / self.joint_limits[2]
        return  joint_angles

    def get_joint_vels(self, normalised):
        if normalised:
            joint_vels = np.array([np.clip(i[1] / self.max_joint_vel, -1, 1) for i in p.getJointStates(self.dog, self.joint_ids, physicsClientId=self.client)])
        else:
            joint_vels = [
                p.getJointState(self.dog, motor_id, self.client)[1]
                for motor_id in self.joint_ids
            ]
        return joint_vels

    def get_observation(self, plane):
        pos, ang = p.getBasePositionAndOrientation(self.dog, self.client)
        # angs = tuple(i / np.pi for i in p.getEulerFromQuaternion(ang))  # normalise base angle
        # lin_vel, ang_vel = self.norm_base_vels()
        # observation = np.array(pos + angs + lin_vel + ang_vel,  dtype=np.float32)
        joint_angles = self.get_joint_angs(normalised=False)  # normalise joint pos
        joint_vels = self.get_joint_vels(normalised=False)
        # foot_contacts = self.detect_foot_contacts_binary(ground_id=plane)
        # observation = np.concatenate((np.array(angs + joint_angles + joint_vels), lin_vel, ang_vel, foot_contacts), axis=0).astype(np.float32)
        observation = np.concatenate((joint_angles, joint_vels , self.joint_torques, ang))
        return observation

    def get_motor_torques_and_vels(self):
        motor_vels = [
            p.getJointState(self.dog, motor_id, self.client)[1]
            for motor_id in self.joint_ids
        ]
        motor_torques = [
            p.getJointState(self.dog, motor_id, self.client)[3]
            for motor_id in self.joint_ids
        ]
        return motor_vels, motor_torques

    def detect_foot_contacts_binary(self, ground_id):
        contact_points = p.getContactPoints(bodyA=self.dog, bodyB=ground_id)
        foot_contacts_binary = [0] * len(self.foot_link_ids)  # Initialize all feet as 0 (no contact)
        for point in contact_points:
            link_index = point[3]  # Get the link index of the robot in contact
            if link_index in self.foot_link_ids:
                foot_index = self.foot_link_ids.index(link_index)  # Get the index of the foot in the foot_link_ids list
                foot_contacts_binary[foot_index] = 1  # Set to 1 if contact is detected
        return np.array(foot_contacts_binary)

    def check_foot_friction(self):
        for foot_link in self.foot_link_ids:
            dynamics_info = p.getDynamicsInfo(self.dog, foot_link)
            friction = dynamics_info[1]  # Lateral friction is the second item in the tuple
            print(f"Foot link {foot_link} has friction: {friction}")

    def set_foot_friction(self, foot_friction=1):
        for foot_link in self.foot_link_ids:
            p.changeDynamics(bodyUniqueId=self.dog, linkIndex=foot_link, lateralFriction=foot_friction,
                             physicsClientId=self.client)

    def set_base_friction(self, base_friction=1):
        p.changeDynamics(bodyUniqueId=self.dog, linkIndex=0, lateralFriction=base_friction,
                             physicsClientId=self.client)

    def reset_init_joints(self):
        self.init_joint_angles
        for i, joint_index in enumerate(self.joint_ids):
            # Example: Set each joint to a specific initial position
            p.resetJointState(self.dog, joint_index, self.init_joint_angles[i], targetVelocity=0.0, physicsClientId=self.client)

    def apply_external_force(self):
        mag = np.random.uniform(0, 100)  # max force of 100N
        th = np.random.uniform(-3.14, 3.14)
        force = [mag * np.cos(th), mag * np.cos(th), 0]
        p.applyExternalForce(objectUniqueId=self.dog,
                             linkIndex=-1,
                             forceObj=force,
                             posObj=[0, 0, 0],
                             flags=p.WORLD_FRAME)
        # print(force)
