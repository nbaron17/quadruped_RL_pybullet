import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import dog_walking
import os
import sys
import time
import cv2
from dog_walking.envs.dog_walking_env import DogWalkingEnv
import matplotlib.pyplot as plt

MODEL_PATH = "models18/rl_model_58560000_steps.zip"
ACTIONS_PATH = "recorded_actions"


def record_actions():
    env = gym.make('dogWalkingEnv-v0', render_mode='human')
    model = PPO.load(MODEL_PATH, env=env)
    env.reset()
    obs = env.reset()[0]
    just_in = True
    for _ in range(5000):
        action, _ = model.predict(obs)
        if just_in:
            concatenated_array = env.dog.get_norm_joint_angs().copy().reshape(12, 1)
            just_in = False
        # action = env.get_trot_gait()
        # action = np.array(12*[0])
        vals = env.dog.get_norm_joint_angs().copy().reshape(12, 1)
        concatenated_array = np.hstack((concatenated_array, vals))
        obs, reward, done, _, _ = env.step(action)
    env.close()
    np.save(f'{ACTIONS_PATH}/joint_values1.npy', concatenated_array)


if __name__=="__main__":
    record_actions()

