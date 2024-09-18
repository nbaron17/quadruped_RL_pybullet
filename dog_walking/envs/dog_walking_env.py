import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
from dog_walking.resources.dog import Dog
import matplotlib.pyplot as plt
from gymnasium import spaces
from collections import deque
from ..resources.gaitPlanner import trotGait
from ..resources.kinematic_model import robotKinematics


class DogWalkingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 60}

    def __init__(self, render_mode=None, fixed_base=False, start_height=0.2):
        super(DogWalkingEnv, self).__init__()
        self.render_mode = render_mode
        # action space is 0 to 2.5, here it has been normalised
        self.action_space = spaces.Box(
            low=np.array([-1.0]*12, dtype=np.float32),
            high=np.array([1.0]*12, dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.array(37 * [-1], dtype=np.float32),
            high=np.array(37 * [1], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        # self.client = p.connect(p.DIRECT)
        if render_mode in ['human', 'rgb_array']:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)


        # Reduce length of episodes for RL algorithms
#        p.setTimeStep(1/30, self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        self.dog = None
        self.plane = None
        self.fixed_base = fixed_base
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.max_steps = 5000
        self.oris = None
        # self.time_step = p.getPhysicsEngineParameters()['fixedTimeStep']
        self.time_step = 0.01
        p.setTimeStep(self.time_step)
        # self.motor_latency = 10  # 10 time steps = 0.1 seconds delay
        # self.command_queue = deque(maxlen=self.motor_latency)
        # no_of_actions = 9
        # for _ in range(self.motor_latency):
        #     self.command_queue.append(np.array(no_of_actions*[0]))

        # Camera parameters
        # self.camera_width = 320
        # self.camera_height = 240
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_fov = 60
        self.camera_aspect = self.camera_width / self.camera_height
        self.camera_near = 0.1
        self.camera_far = 100.0

        self.trot = trotGait()
        # foot separation (Ydist = 0.16 -> tetta=0) and distance to floor
        Xdist, Ydist, height = 0.20, 0.15, 0.15
        T = 0.5  # period of time (in seconds) of every step
        self.offset = np.array([0.5, 0., 0., 0.5])  # defines offset between each footstep in this order (FR,FL,BR,BL)
        self.bodytoFeet0 = np.matrix([[Xdist / 2, -Ydist / 2, -height],
                                      [Xdist / 2, Ydist / 2, -height],
                                      [-Xdist / 2, -Ydist / 2, -height],
                                      [-Xdist / 2, Ydist / 2, -height]])
        self.robotKinematics = robotKinematics()
        # self.obs_stack = deque(maxlen=3)

        self.history_length = 50  # 0.5 seconds action history
        # init_action_history = [[0, 0, 0]] * int(history_length / 2) + [[1, 1, 1]] * int(history_length / 2)
        # if initialised all to same number, then would get penalty straight away
        # self.FR_angles_history = deque(init_action_history, maxlen=history_length)
        # self.FL_angles_history = deque(init_action_history, maxlen=history_length)
        # self.BL_angles_history = deque(init_action_history, maxlen=history_length)
        # self.BR_angles_history = deque(init_action_history, maxlen=history_length)

        self.action_history = deque(maxlen=self.history_length)
        self.vel_sign_history = deque(maxlen=self.history_length)

        self.reset(start_height=start_height)

    # def update_action_history(self, action):
    #     self.FR_angles_history.append([action[0], action[1], action[2]])
    #     self.FL_angles_history.append([action[3], action[4], action[5]])
    #     self.BL_angles_history.append([action[6], action[7], action[8]])
    #     self.BR_angles_history.append([action[9], action[10], action[11]])


    def step(self, action):
        # Feed action to the dog and get observation of dog's state
        # self.command_queue.append(action)
        # delayed_action = self.command_queue.popleft()
        self.dog.apply_action(action)
        # print(self.client)
        p.stepSimulation(physicsClientId=self.client)
        ob = self.dog.get_observation(self.plane)
        self.action_history.append(self.dog.get_norm_joint_angs().copy())
        # self.update_action_history(action)

        reward = self.reward_function()
        
        self.step_cntr += 1

        self.oris = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.dog.dog, self.client)[1])
        _ = self.check_if_done()
        # if np.absolute(self.oris[0])>1.6 or np.absolute(self.oris[1])>1.6:
        #     reward -= 1.
        #     self.acc_reward += reward

        # self.obs_stack.append(ob)
        # flat_obs_stack = np.concatenate(self.obs_stack)

        return ob, reward, self.done, False, dict()

    def check_if_done(self):
        if self.step_cntr > self.max_steps:  # or np.absolute(self.oris[0]) > 1.6 or np.absolute(self.oris[1]) > 1.6:
            self.done = True
        return self.done

    def body_contact_penalty(self):
        contact_points = p.getContactPoints(self.plane, self.dog.dog, -1, -1)
        reward = - 0.01 if len(contact_points) > 0 else 0
        # print('body contact: ', reward)
        return reward

    def tilting_penalty(self):
        # oris = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.dog.dog, self.client)[1])
        # max_tilt = max(np.absolute(oris[0]), np.absolute(oris[1]))
        # tilt_th = 0.78  # 45-degree tilt threshold
        # if max_tilt > tilt_th:
        #     return - 0.001 * (max_tilt - tilt_th) / tilt_th
        # else:
        #     return 0
        # Penalty for sideways rotation of the body.
        orientation = p.getBasePositionAndOrientation(self.dog.dog, self.client)[1]
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        scaler = 0.001
        reward = scaler * -np.absolute(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # print('tilt: ', reward)
        return reward

    def yaw_penalty(self):
        # Disincentivise not looking forwards
        oris = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.dog.dog, self.client)[1])
        yaw = np.absolute(oris[2])
        yaw_th = 0.52  # 30-degree yaw threshold
        if yaw > yaw_th:
            reward = - 0.001 * (yaw - yaw_th) / yaw_th
            # print('yaw: ', reward)
            return reward
        else:
            # print('yaw: ', 0)
            return 0

    def drift_penalty(self, base_state):
        reward = -np.absolute(base_state[0][1] - self.prev_pos[1])
        # print('drift: ', reward)
        return reward

    def energy_penalty(self):
        vels, torques = self.dog.get_motor_torques_and_vels()
        scaler = 0.0005
        energy =  scaler * -np.absolute(np.dot(vels, torques)) * self.time_step
        # print('energy: ', energy)
        return energy

    def vibrating_leg_penalty(self):
        # for each leg, at least one of the joint has to have moved more than a certain threshold during the recent
        # action history, otherwise a penalty is incurred
        threshold = 0.01
        for history in [self.FR_angles_history, self.FL_angles_history, self.BL_angles_history, self.BR_angles_history]:
            joint1 = [item[0] for item in history]
            joint2 = [item[1] for item in history]
            joint3 = [item[2] for item in history]
            range1 = max(joint1) - min(joint1)
            range2 = max(joint2) - min(joint2)
            range3 = max(joint3) - min(joint3)
            if max([range1, range2, range3]) < threshold:
                return - 1
        return 0

    def vibrating_leg_penalty_2(self):
        if len(self.action_history) == self.history_length:
            std_dev = np.std(self.action_history, axis=0)
            return std_dev
        else:
            return np.array(12*[1.0])
            # MOVEMENT = 0.1
            # no_movement_count = 0
            # for joint in std_dev:
            #     if joint < MOVEMENT:
            #         no_movement_count += 1
            #
            # if no_movement_count >= 5:
            #     std_dev = np.round(std_dev, decimals=4)
            #     reward = -1

    def vibrating_leg_penalty3(self):
        # Check if joint velocities have changed direction > twice per half second
        joint_vel_signs = np.array([np.sign(i[1]) for i in p.getJointStates(self.dog.dog, self.dog.joint_ids, physicsClientId=self.client)])
        self.vel_sign_history.append(joint_vel_signs)
        vel_hist_array = np.array(list(self.vel_sign_history)).T  # 12 x 50 array
        for col in vel_hist_array:
            change_count = 0
            for i in range(1, len(col)):
                if (col[i - 1] == 1 and col[i] == -1) or (col[i - 1] == -1 and col[i] == 1):
                    change_count += 1
            if change_count > 2:
                return - 0.01
        return 0


    def reward_function(self):
        base_state = p.getBasePositionAndOrientation(self.dog.dog, self.client)
#        reward = np.linalg.norm(np.array(base_state[0][0:2]) - self.prev_pos) 
        reward = base_state[0][0] - self.prev_pos[0] # Reward now the movement along the x-axis
        # reward += self.drift_penalty(base_state)
        # reward += self.tilting_penalty()
        reward += self.body_contact_penalty()
        # reward += self.energy_penalty()
        # reward += self.yaw_penalty()
        reward += self.vibrating_leg_penalty3()
        self.acc_reward += reward
        self.prev_pos = np.array(p.getBasePositionAndOrientation(self.dog.dog, self.client)[0][0:2]) # x-y base position
        # if reward == 0.0:
        #     print(self.client, self.dog.dog, base_state[0][0])
        return reward

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None, start_height=0.2):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        # Reload the plane and dog

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.dog = Dog(client=self.client, fixed_base=self.fixed_base, start_height=start_height)

        x = (np.random.uniform(5, 9) if np.random.randint(2) else
             np.random.uniform(-5, -9))
        y = (np.random.uniform(5, 9) if np.random.randint(2) else
             np.random.uniform(-5, -9))
        self.done = False
        self.step_cntr = 0
        self.acc_reward = 0.

        # Get observation to return
        ob = self.dog.get_observation(self.plane)

        init_state = p.getBasePositionAndOrientation(self.dog.dog, self.client)
        self.prev_pos = np.array(init_state[0][0:2]) # x-y base position
        info = dict()
        self.oris = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.dog.dog, self.client)[1])


        return (np.array(ob, dtype=np.float32), info)

    def render(self, mode='human'):
        if self.render_mode == "rgb_array":
            return self._render_frame()

        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        dog_id, client_id = self.dog.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(dog_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
#        self.rendered_img.set_data(frame)
#        plt.draw()
#        plt.pause(.00001)

    def _render_frame(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.dog.dog)

        # Adjust the camera to follow the robot
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=robot_pos,
            distance=3,  # Adjust distance as needed
            yaw=0,  # Adjust yaw for desired view angle
            pitch=-30,  # Adjust pitch for desired view angle
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        img_arr = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # Better quality renderer
        )
        rgb_array = np.reshape(img_arr[2], (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        return rgb_array

    def get_trot_gait(self):
        L, angle, Lrot, T = 0.6, 0.0, 0.0, 0.8
        pos, orn = np.zeros([3]), np.zeros([3])
        bodytoFeet = self.trot.loop(L, angle, Lrot, T, self.offset, self.bodytoFeet0)
        FR_angles, FL_angles, BR_angles, BL_angles, transformedBodytoFeet = self.robotKinematics.solve(orn, pos, bodytoFeet)
        action = np.concatenate((FR_angles, FL_angles, BR_angles, BL_angles)) / 3.14
        return action


    def close(self):
        p.disconnect(self.client)