from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if there are completed episodes in the buffer
        if len(self.model.ep_info_buffer) > 0:
            # Retrieve the last episode's reward
            last_episode_reward = self.model.ep_info_buffer[-1]['r']
            self.logger.record('episode_reward', last_episode_reward)

        return True

