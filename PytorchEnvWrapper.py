from types import SimpleNamespace

import gym
import torch

class PytorchWrapper(gym.Wrapper):
    '''
    gym wrapper returning pytorch tensors instead of numpy array
    '''
    def __init__(self, env, dtype = torch.float):
        super().__init__(env)
        self.dtype = dtype
        self.action_space = self._getSpaceData(env.action_space)
        self.observation_space = self._getSpaceData(env.observation_space)

    def _getSpaceData(self, space):
        isDiscrete = isinstance(space, gym.spaces.Discrete)
        sample = space.sample

        if isDiscrete:
            shape = (space.n,)
            low = torch.zeros(space.n, 1)
            high = torch.ones(space.n, 1)
        else:
            shape = space.shape
            low = torch.tensor(space.low, dtype=self.dtype)
            high = torch.tensor(space.high, dtype=self.dtype)

        return SimpleNamespace(isDiscrete=isDiscrete, shape=shape, low=low, high=high, sample=sample)

    def reset(self):
        return torch.tensor(self.env.reset(), dtype=self.dtype)

    def step(self, actionTensor):
        obs, reward, done, info = self.env.step(actionTensor.numpy())
        
        return torch.tensor(obs, dtype=self.dtype), torch.tensor(reward, dtype=self.dtype), done, info
