#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:51:40 2020

@author: linux-asd
"""

import pybullet as p
import numpy as np
import time
import pybullet_data
from dog_walking.resources.pybullet_debuger import pybulletDebug
from dog_walking.resources.kinematic_model import robotKinematics
from dog_walking.resources.gaitPlanner import trotGait
from dog_walking.resources.sim_fb import systemStateEstimator


def detect_foot_contacts_binary(robot_id, foot_link_ids, ground_id):
    contact_points = p.getContactPoints(bodyA=robot_id, bodyB=ground_id)

    foot_contacts_binary = {link_id: 0 for link_id in foot_link_ids}  # Initialize all feet as 0 (no contact)

    for point in contact_points:
        link_index = point[3]  # Get the link index of the robot in contact
        if link_index in foot_link_ids:
            foot_contacts_binary[link_index] = 1  # Set to 1 if contact is detected

    return foot_contacts_binary

def run():
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)

    cubeStartPos = [0, 0, 0.2]
    FixedBase = False  # if fixed no plane is imported
    if (FixedBase == False):
        ground = p.loadURDF("plane.urdf")
    boxId = p.loadURDF("dog_walking/resources/dog.urdf", cubeStartPos, useFixedBase=FixedBase)

    jointIds = []
    paramIds = []
    time.sleep(0.5)
    for j in range(p.getNumJoints(boxId)):
        #    p.changeDynamics(boxId, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(boxId, j)
        print(info)
        jointName = info[1]
        jointType = info[2]
        jointIds.append(j)

    footFR_index = 3
    footFL_index = 7
    footBR_index = 11
    footBL_index = 15

    pybullet_debug = pybulletDebug()
    robot_kinematics = robotKinematics()
    trot = trotGait()
    meassure = systemStateEstimator(boxId)  # meassure from simulation

    # robot properties
    maxForce = 200  # N/m
    masVel = 3.703  # rad/s
    """initial foot position"""
    # foot separation (Ydist = 0.16 -> tetta=0) and distance to floor
    Xdist = 0.20
    Ydist = 0.15
    height = 0.15
    # body frame to foot frame vector
    bodytoFeet0 = np.matrix([[Xdist / 2, -Ydist / 2, -height],
                             [Xdist / 2, Ydist / 2, -height],
                             [-Xdist / 2, -Ydist / 2, -height],
                             [-Xdist / 2, Ydist / 2, -height]])

    T = 0.5  # period of time (in seconds) of every step
    offset = np.array([0.5, 0., 0., 0.5])  # defines the offset between each foot step in this order (FR,FL,BR,BL)
    bodytoFeet_vecX = 0.
    bodytoFeet_vecY = 0.

    pos = np.zeros([3])
    orn = np.zeros([3])
    p.setRealTimeSimulation(1)
    p.setTimeStep(0.002)

    for i in range(100000):
        lastTime = time.time()

        _, _, L, angle, Lrot, T = pybullet_debug.cam_and_robotstates(boxId)
        # calculates the feet coord for gait, defining length of the step and direction (0º -> forward; 180º -> backward)
        # print(L, angle, Lrot, T)
        bodytoFeet = trot.loop(L, angle, Lrot, T, offset, bodytoFeet0)

        #####################################################################################
        #####   kinematics Model: Input body orientation, deviation and foot position    ####
        #####   and get the angles, neccesary to reach that position, for every joint    ####
        FR_angles, FL_angles, BR_angles, BL_angles, transformedBodytoFeet = robot_kinematics.solve(orn, pos, bodytoFeet)
        # FR_angles[0], FL_angles[0], BR_angles[0], BL_angles[0] = 0.5, 0.5, 0.5, 0.5

        t, X = meassure.states()
        U, Ui, torque = meassure.controls()
        bodytoFeet_vecX = np.append(bodytoFeet_vecX, bodytoFeet[0, 0])
        bodytoFeet_vecY = np.append(bodytoFeet_vecY, bodytoFeet[0, 2])

        # move movable joints
        for i in range(3):
            p.setJointMotorControl2(boxId, i, p.POSITION_CONTROL,
                                    targetPosition=FR_angles[i], force=maxForce, maxVelocity=masVel)
            p.setJointMotorControl2(boxId, 4 + i, p.POSITION_CONTROL,
                                    targetPosition=FL_angles[i], force=maxForce, maxVelocity=masVel)
            p.setJointMotorControl2(boxId, 8 + i, p.POSITION_CONTROL,
                                    targetPosition=BR_angles[i], force=maxForce, maxVelocity=masVel)
            p.setJointMotorControl2(boxId, 12 + i, p.POSITION_CONTROL,
                                    targetPosition=BL_angles[i], force=maxForce, maxVelocity=masVel)

        foot_link_ids = [3, 7, 11, 15]  # FR, FL, BR, BL
        foot_contacts = detect_foot_contacts_binary(robot_id=boxId, foot_link_ids=foot_link_ids, ground_id=ground)

        # joint_ids = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        # print('here', [
        #     p.getJointState(boxId, motor_id)[1:4]
        #     for motor_id in joint_ids
        # ][0])
        # print(motor_vels, motor_torques)

    #    print(time.time() - lastTime)
    p.disconnect()

if __name__=="__main__":
    run()