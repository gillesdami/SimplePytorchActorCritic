import gym

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from Model import Policy
from PytorchEnvWrapper import PytorchWrapper
from Util import discountRewards, runPolicy

def train(params, cb = lambda loss, reward_mean, reward_sum: None):
    writer = SummaryWriter(params.log_dir)
    writer.add_text('params', str(params), 0)

    env = PytorchWrapper(gym.make(params.env))
    
    env.seed(params.seed)
    torch.manual_seed(params.seed)

    model = Policy()
    opt = optim.Adam(model.parameters(), lr=params.lr)

    def learn(expBuffer, i):
        action_log_probs, rewards, prewards = list(map(torch.stack, zip(*expBuffer)))

        drewards = discountRewards(rewards, params.gamma)
        drewards_norm = (drewards - drewards.mean()) / (drewards.std() + 1e-8)
        advantages = drewards_norm - prewards
        
        policy_loss = torch.sum(-action_log_probs * advantages)
        critic_loss = F.smooth_l1_loss(prewards, drewards_norm, reduction='sum')
        
        opt.zero_grad()
        loss = policy_loss + critic_loss
        loss.backward()
        opt.step()
        
        writer.add_scalar('reward/avg', rewards.mean(), i)
        writer.add_scalar('reward/sum', rewards.sum(), i)
        writer.add_scalar('loss/policy', policy_loss, i)
        writer.add_scalar('loss/critic', critic_loss, i)
        writer.add_scalar('loss/loss', loss, i)
        cb(loss, rewards.mean(), rewards.sum())

    def main():
        for i in range(params.episode):
            obs = env.reset()
            expBuffer = []

            for step in range(params.max_step):
                action_probs, preward = model(obs)
        
                m = Categorical(action_probs)
                action = m.sample()

                obs, reward, done, _ = env.step(action)
                
                expBuffer.append((m.log_prob(action), reward, preward))

                if done:
                    break

            learn(expBuffer, i)
            print('Episode {}\tLast length: {:5d}'.format(i, step))
        torch.save(model.state_dict(), params.log_dir + '/finalStateDict')
        
    main()
    