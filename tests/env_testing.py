import gymnasium as gym
import numpy as np
import time
from dog_walking.envs.dog_walking_env import DogWalkingEnv
import numpy as np
from dog_walking.resources.dog import Dog


def test_std_dev_for_leg_vibration(env):
    action = np.array(12 * [0.0])
    i = 0
    count = 0
    while True:
        i += 1
        if i % 10 == 0:
            count += 1
        if count % 2 == 0:
            action[0:3] = 0.2
        else:
            action[0:3] = -0.2
        obs, reward, done, _, _ = env.step(action)
        print(env.vibrating_leg_penalty_2())

def test_std_dev_for_trot_gait(env):
    just_in = True
    for _ in range(2000):
        action = env.get_trot_gait()
        obs, reward, done, _, _ = env.step(action)
        if just_in:
            vels, _ = env.dog.get_motor_torques_and_vels()
            concatenated_array = np.array(vels).reshape(12, 1)
            just_in = False
            continue
        vels, _ = env.dog.get_motor_torques_and_vels()
        concatenated_array = np.hstack((concatenated_array, np.array(vels).reshape(12, 1)))
    env.close()
    np.save('recorded_actions/trot_vels.npy', concatenated_array)

def test_trot_gait(env):
    while True:
        action = env.get_trot_gait()
        obs, reward, done, _, _ = env.step(action)

def test_joint_sampling(env):
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        time.sleep(0.1)

def test_joint_control(env):
    init_time = time.time()
    while True:
        val = np.sin(time.time() - init_time)
        action = np.array([0, val, val, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        obs, reward, done, _, _ = env.step(action)

def test_joint_limits(env):
    init_time = time.time()
    val = 10*[0.0]
    i = 0
    while True:
        # action = np.array([val[i], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action = np.array(12*[0.0])
        obs, reward, done, _, _ = env.step(action)
        # print(np.round(env.dog.get_norm_joint_angs(), 1))
        if time.time()-init_time > 5:
            i+=1
            init_time = time.time()

def test_init_joint_angles(env):
    obs, reward, done, _, _ = env.step(np.array(12*[0.0]))
    action = np.array([0.0,0.0,0.0001, 0.0, 0.0 ,0.0 ,0.0, 0.0 ,0.0, 0.0, 0.0, 0.0])
    while True:
        obs, reward, done, _, _ = env.step(action)


def test_motor_torques():
    import pybullet as p
    import pybullet_data
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    plane = p.loadURDF("dog_walking/resources/simpleplane.urdf", physicsClientId=client)
    dog = Dog(client=client, fixed_base=True, start_height=0.5)
    time.sleep(1)
    p.changeDynamics(dog.dog, 0, jointDamping=0.1, lateralFriction=0.1)
    p.setJointMotorControl2(dog.dog, 0, p.POSITION_CONTROL,
                            targetPosition = 1.0 , force = 100 , maxVelocity = 100, physicsClientId=client)
    p.setJointMotorControl2(bodyIndex=dog.dog, jointIndex=1,
                            controlMode=p.TORQUE_CONTROL,
                            force=.1, physicsClientId=client)
    while True:
        p.stepSimulation()

def get_link_ids(env):
    env.dog.get_link_names_and_ids()


def main():
    env = gym.make('dogWalkingEnv-v0', render_mode='human', fixed_base=False, start_height=0.2, no_of_actions=12)
    obs = env.reset(start_height=0.2)[0]
    # test_motor_torques()
    while True:
        # test_joint_limits(env)
        # test_joint_control(env)
        test_joint_sampling(env)
        # test_trot_gait(env)
        # test_trot_gait(env)
        # test_init_joint_angles(env)

if __name__=="__main__":
    main()