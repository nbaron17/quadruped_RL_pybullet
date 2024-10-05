import gymnasium as gym
from stable_baselines3 import PPO, SAC
import os
# import cv2
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from dog_walking.utils.tensorboard_callback import TensorboardCallback

# Create directories to hold models and logs
model_dir = "../models/models30"
log_dir = "../logs/training_logs/logs30"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def make_env(rank, seed=0):
    def _init():
        env = gym.make('dogWalkingEnv-v0', no_of_actions=12, ext_force_prob=0.0)
        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def train(env, save_path):
    TIMESTEPS = 100000000
    save_steps = 200000  # multiplied by number of envs to get save freq
    checkpoint_callback = CheckpointCallback(save_freq=save_steps, save_path=save_path)
    callbacks = [checkpoint_callback, TensorboardCallback()]
    # model = PPO.load("models29/final2.zip", env=env)
    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir) #, batch_size=128, n_steps=170, learning_rate=0.0001)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, callback=CallbackList(callbacks))
    model.save(f"{save_path}/final")

def test(env, path_to_model):
    model = PPO.load(path_to_model, env=env)
    env.reset()
    obs = env.reset()[0]
    just_in = True
    for _ in range(3000):
        action, _ = model.predict(obs)
        if just_in:
            # vels, _ = env.dog.get_motor_torques_and_vels()
            # concatenated_array = np.array(vels).reshape(12, 1)
            just_in = False
        vib = env.vibrating_leg_penalty3()
        # concatenated_array = np.hstack((concatenated_array, np.array(vels).reshape(12, 1)))
        obs, reward, done, _, _ = env.step(action)
        # print(env.unwrapped.acc_reward)
        # time.sleep(0.1)
        if done:
            print(f"Total Reward: {env.unwrapped.acc_reward}")
            break
    env.close()
    # np.save('recorded_actions/vel_values.npy', concatenated_array)

if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()
    # python train.py dogWalkingEnv-v0 PPO -t
    # python train.py dogWalkingEnv-v0 PPO --test ./models/SAC_2150000.zip


    if args.train:
        # gymenv = gym.make('dogWalkingEnv-v0')
        # env = make_vec_env(gymenv, n_envs=1, env_kwargs=dict(grid_size=10))
        # env = gym.make('dogWalkingEnv-v0')
        # env = Monitor(gym.make("dogWalkingEnv-v0"))
        # env = DummyVecEnv([lambda: Monitor(gym.make("dogWalkingEnv-v0"))])  # this one
        # env = DummyVecEnv([lambda: gym.make('dogWalkingEnv-v0')])
        num_cpu = 12  # Also sets the number of episodes per training iteration
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        train(env, save_path=model_dir)

    if(args.test):
        if os.path.isfile(args.test):
            # gymenv = gym.make(args.gymenv, render_mode='human')
            # gymenv = gym.make(args.gymenv)
            # unique_prefix = "model29-final2_" + time.strftime("%Y%m%d-%H%M%S")
            # gymenv = RecordVideo(gym.make('dogWalkingEnv-v0',  render_mode='rgb_array', no_of_actions=12), 'videos', name_prefix=f"rl-video-{unique_prefix}", episode_trigger=lambda x: True)
            gymenv = gym.make('dogWalkingEnv-v0', render_mode='human', no_of_actions=12, ext_force_prob=0.005)
            # gymenv = DummyVecEnv([lambda: gym.make('dogWalkingEnv-v0')])
            test(gymenv, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')