import d4rl
import numpy as np


class d4rl_dataset():
    def __init__(self,env):
        dataset = d4rl.qlearning_dataset(env)
        self.dataset = dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
        )
        self.len = self.dataset['observations'].shape[0]

    def get_data(self,batch_size=256):
        idx = np.random.choice(self.len, batch_size)
        return self.dataset['observations'][idx], self.dataset['actions'][idx], self.dataset['rewards'][idx], self.dataset['next_observations'][idx], self.dataset['dones'][idx]