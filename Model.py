import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    '''
    Simple ActorCritic model for discrete action spaces
    '''
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        reward_value = self.value_head(x)
        return F.softmax(action_scores, dim=-1), reward_value[0]