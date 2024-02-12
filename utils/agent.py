import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(in_features=148, # 147 – flatten image, 1 – direction
                      out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(in_features=120,
                      out_features=60),
            nn.ReLU(),
            nn.BatchNorm1d(60),
            nn.Linear(in_features=60,
                      out_features=16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16,
                      out_features=1)
        )
        self.actor = nn.Sequential(
            nn.Linear(in_features=148, # 147 – flatten image, 1 – direction
                      out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(in_features=120,
                      out_features=60),
            nn.ReLU(),
            nn.BatchNorm1d(60),
            nn.Linear(in_features=60,
                      out_features=16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16,
                      out_features=envs.single_action_space.n)
        )
    def get_value(self, x):
        return self.critic(x.permute(0,3,1,2))

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x.permute(0,3,1,2))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x.permute(0,3,1,2))

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# class Agent(nn.Module):
#     def __init__(self, envs=" "):
#         super().__init__()
#         self.critic = nn.Sequential(
#                                    layer_init(nn.Conv2d(in_channels=3,
#                                                         out_channels=16,
#                                                         kernel_size=3,
#                                                         padding=1)),
#                                    nn.ReLU(),
#                                    nn.BatchNorm2d(16),
#                                    layer_init(nn.Conv2d(in_channels=16,
#                                                         out_channels=3,
#                                                         kernel_size=3,
#                                                         padding=1)),
#                                    nn.ReLU(),
#                                    nn.BatchNorm2d(3),

#                                    layer_init(nn.Conv2d(in_channels=3,
#                                                         out_channels=1,
#                                                         kernel_size=3,
#                                                         padding=1)),

#                                    nn.Flatten(),

#                                    layer_init(nn.Linear(49,7)),
#                                    nn.ReLU(),

#                                    layer_init(nn.Linear(7,1))
#                                    )

#         self.actor = nn.Sequential(
#                                   layer_init(nn.Conv2d(in_channels=3,
#                                                         out_channels=16,
#                                                         kernel_size=3,
#                                                         padding=1)),
#                                    nn.ReLU(),
#                                    nn.BatchNorm2d(16),
#                                    layer_init(nn.Conv2d(in_channels=16,
#                                                         out_channels=3,
#                                                         kernel_size=3,
#                                                         padding=1)),
#                                    nn.ReLU(),
#                                    nn.BatchNorm2d(3),

#                                    layer_init(nn.Conv2d(in_channels=3,
#                                                         out_channels=1,
#                                                         kernel_size=3,
#                                                         padding=1)),

#                                    nn.Flatten(),

#                                    layer_init(nn.Linear(49,7)),
#                                    nn.ReLU(),

#                                    layer_init(nn.Linear(7,3))
#                                    )

#     def get_value(self, x):
#         return self.critic(x.permute(0,3,1,2))

#     def get_action_and_value(self, x, action=None):
#         logits = self.actor(x.permute(0,3,1,2))
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x.permute(0,3,1,2))
