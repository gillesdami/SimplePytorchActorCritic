import argparse

from train import train

parser = argparse.ArgumentParser(description='PyTorch actor-critic')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--env', default='CartPole-v0', help='gym env name (default: CartPole-v0)')
parser.add_argument('--episode', type=int, default=1000, help='epidose count (default: 1000)')
parser.add_argument('--max-step', type=int, default=10000, help='max step per epidose (default: 10000)')
parser.add_argument('--lr', type=float, default=3e-2, help='learning rate (default: 3e-2)')
parser.add_argument('--log-dir', default='logs', help='tensorboard and model logging directory (default: logs)')

params = parser.parse_args()

train(params)