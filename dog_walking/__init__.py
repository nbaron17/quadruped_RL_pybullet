from gymnasium.envs.registration import register
register(
    id='dogWalkingEnv-v0',
    entry_point='dog_walking.envs:DogWalkingEnv'
)
