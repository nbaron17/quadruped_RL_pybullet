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



def enjoy_agent(model_path, n_episodes=5, record_video=False):
    """
    Load a trained agent and run it in the environment to enjoy its behavior.

    :param model_path: Path to the saved model.
    :param n_episodes: Number of episodes to run the agent.
    """
    if record_video:
        from gymnasium.wrappers import RecordVideo
        import time
        os.makedirs('videos', exist_ok=True)
        n_episodes = 1
        unique_prefix = time.strftime("%Y%m%d-%H%M%S")
        env = RecordVideo(gym.make('dogWalkingEnv-v0',  render_mode='rgb_array', no_of_actions=12), 'videos', name_prefix=f"rl-video-{unique_prefix}", episode_trigger=lambda x: True)
    else:
        env = gym.make('dogWalkingEnv-v0', render_mode='human', no_of_actions=12, ext_force_prob=0.0)

    # Load the trained model
    model = PPO.load(model_path)  # Replace PPO with your RL algorithm

    # Run the agent in the environment for `n_episodes`
    for episode in range(n_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run a trained RL agent in the environment")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--n_episodes', type=int, default=5, help='Number of episodes to run the agent')
    parser.add_argument('--record_video', type=bool, default=False, help='True to record a video and set n_episodes as 1.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enjoy_agent(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        record_video=args.record_video
    )
