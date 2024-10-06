import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
import gymnasium as gym
from stable_baselines3 import PPO
import argparse
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import dog_walking


def make_env(rank, seed=0):
    def _init():
        env = gym.make('dogWalkingEnv-v0', no_of_actions=12, ext_force_prob=0.0)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def evaluate_agent(model_path, render=True, n_eval_episodes=10):
    render_mode = 'human' if render else None
    env = gym.make('dogWalkingEnv-v0', render_mode=render_mode, no_of_actions=12, ext_force_prob=0.0)
    model = PPO.load(model_path, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=render)
    print(f"Mean reward: {mean_reward} +/- {std_reward} after {n_eval_episodes} episodes")

    env.close()


def parse_args():
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true', default=True, help='Render the environment during evaluation')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate_agent(
        model_path=args.model_path,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render
    )

