import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import gymnasium as gym
from stable_baselines3 import PPO, SAC
# import cv2
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from dog_walking.utils.tensorboard_callback import TensorboardCallback
import yaml


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'dog_walking', 'config', 'config.yaml')


def make_env(rank, config, seed=0):
    def _init():
        env = gym.make('dogWalkingEnv-v0', no_of_actions=config['no_of_actions'], ext_force_prob=config['ext_force_prob'])
        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def train_agent(config_path=DEFAULT_CONFIG_PATH, load_model_path=None):
    """
    Train a reinforcement learning agent based on a configuration file.

    :param config_path: Path to the YAML config file containing training parameters.
    """
    # Load the config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    env = SubprocVecEnv([make_env(i, config) for i in range(config['num_cpu'])])

    # Check if we are loading a pre-trained model or starting fresh
    algo = config['algorithm']
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading existing model from {load_model_path}")
        model = PPO.load(load_model_path, env=env)  # Replace PPO with your algorithm
    else:
        print("Training a new model...")
        if algo == "PPO":
            model = PPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=config['tensorboard_log'])

    checkpoint_callback = CheckpointCallback(save_freq=config['save_freq'], save_path='./models/',
                                             name_prefix='rl_model_checkpoint')

    # Optional: Evaluate the model during training
    # eval_env = gym.make(env_id)
    # eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model/',
    #                              log_path='./logs/', eval_freq=config['eval_freq'],
    #                              deterministic=True, render=False)

    model.learn(total_timesteps=config['total_timesteps'], reset_num_timesteps=False, progress_bar=config['progress_bar'],
                callback=[checkpoint_callback, TensorboardCallback()])
    model.save(f"./models/{algo}_final_model")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to the YAML configuration file')
    parser.add_argument('--load_model', type=str, help='Path to an existing model to continue training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_agent(args.config, load_model_path=args.load_model)