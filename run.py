import argparse
import torch
import gym
from torch.distributions import Categorical

from Model import Policy
from PytorchEnvWrapper import PytorchWrapper
from Util import runPolicy

parser = argparse.ArgumentParser(description='Run the model')
parser.add_argument('--env', default='CartPole-v0', help='gym env name (default: CartPole-v0)')
parser.add_argument('--max-episode', type=int, default=1000, help='max epidose count (default: 1000)')
parser.add_argument('--load', default='logs/finalStateDict')
params = parser.parse_args()

env = PytorchWrapper(gym.make(params.env))
model = Policy()
model.load_state_dict(torch.load(params.load))
model.eval()

rewards, step = runPolicy(env, lambda obs: Categorical(model(obs)[0]).sample(), params.max_episode)

print(sum(rewards) / step, step)