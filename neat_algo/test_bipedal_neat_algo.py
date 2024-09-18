import os
import pickle
import gym
import neat
import numpy as np

# load the winner
local_dir = os.path.dirname(__file__)
model_path = os.path.join(local_dir, 'winner-bipedal')
with open(model_path, 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config_cartpole file, which is assumed to live in
# the same directory as this script.
config_path = os.path.join(local_dir, 'config_bipedal')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
env = gym.make("BipedalWalker-v3", render_mode='human')
observation = env.reset()[0]

done = False
while not done:
    action = net.activate(observation)

    observation, reward, done, info, _ = env.step(action)
    env.render()