import torch

def discountRewards(rewards, gamma):
    '''
    return a tensor of discounted reward

    params:
        rewards: tensor of reward
        gamma: discount rate
    '''
    discounted_rewards = []
    reward_sum = 0
    for reward in reversed(rewards):
        reward_sum = reward + gamma * reward_sum
        discounted_rewards.append(reward_sum)

    return reversed(torch.stack(discounted_rewards))

def runPolicy(env, policy, max_step = -1, render = True, logInfo = False):
    '''
    Test a policy

    params:
        env: gym env
        policy: policy function taking an observation and returning an action
        max_step: max step
        render: if to render the environement
    '''
    step = 0
    done = False
    obs = env.reset()
    
    rewards = []
    while not done and step != max_step:
        step += 1
        
        if render:
            env.render()
        obs, reward, done, info = env.step(policy(obs))
        if logInfo:
            print(info)
        rewards.append(reward)

    env.close()
    return rewards, step