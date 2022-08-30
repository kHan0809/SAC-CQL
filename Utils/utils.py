import d4rl
import numpy as np


class d4rl_dataset():
    def __init__(self,env):
        self.max_epi_len = 1000
        dataset = d4rl.qlearning_dataset(env)
        self.dataset = dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
        )
        self.len = self.dataset['observations'].shape[0]
        self.get_Return()

    def get_Return(self):
        Return, curlen = [], 0
        for i in range(self.dataset['observations'].shape[0]):
            if self.dataset['dones'][i] == 1.0 or curlen == self.max_epi_len-1:
                Return.append(sum(self.dataset['rewards'][i - curlen:i + 1]))
                curlen = 0
            else:
                curlen += 1
        Return = np.array(Return)
        print('[Return mean] : ',Return.mean())

    def get_data(self,batch_size=256):
        idx = np.random.choice(self.len, batch_size)
        return self.dataset['observations'][idx], self.dataset['actions'][idx], self.dataset['rewards'][idx], self.dataset['next_observations'][idx], self.dataset['dones'][idx]


