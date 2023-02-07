import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__) + "/..")

from utils import ReplayBuffer

# Deterministic policy network, input s, output the action with n degrees of freedom
class Deterministic_Policy(nn.Module):
    def __init__(self, net_width, obs_count, action_count, usebn=False):
        super(Deterministic_Policy, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.fc1 = nn.Linear(obs_count, net_width)
        self.bn1 = BN(net_width)
        self.fc2 = nn.Linear(net_width, net_width // 2)
        self.bn2 = BN(net_width)
        self.fc3 = nn.Linear(net_width, net_width)
        self.bn3 = BN(net_width)
        self.fc4 = nn.Linear(net_width, net_width)
        self.bn4 = BN(net_width)
        self.fc5 = nn.Linear(net_width // 2, action_count)

    def forward(self, s: torch.Tensor):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.tanh(self.bn3(self.fc3(x)))
        # x = torch.tanh(self.bn4(self.fc4(x)))
        x = torch.tanh(self.fc5(x))
        # x = torch.clamp(self.fc5(x), -1, 1)
        return x

# Stochastic policy network, input s, output the action with n degrees of freedom
class Stochastic_Policy(nn.Module):
    def __init__(self, net_width, obs_count, action_count, usebn=False):
        super(Stochastic_Policy, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.fc1 = nn.Linear(obs_count, net_width)
        self.bn1 = BN(net_width)
        self.fc2 = nn.Linear(net_width, net_width // 2)
        self.bn2 = BN(net_width // 2)
        self.fc3 = nn.Linear(net_width, net_width)
        self.bn3 = BN(net_width)
        self.fc4 = nn.Linear(net_width, net_width)
        self.bn4 = BN(net_width)
        self.fc5 = nn.Linear(net_width // 2, action_count)
        self.fc6 = nn.Linear(net_width // 2, action_count)


    def forward(self, s: torch.Tensor):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.tanh(self.bn3(self.fc3(x)))
        # x = torch.tanh(self.bn4(self.fc4(x)))
        mu = torch.tanh(self.fc5(x))
        std = torch.sigmoid(self.fc6(x))
        # x = torch.clamp(self.fc5(x), -1, 1)
        # print(mu.shape, std.shape)
        return mu, std


# Action Value Q network, input s, output expected total reward
class ActionValue(nn.Module):
    def __init__(self, net_width, obs_count, action_count, usebn=False):
        super(ActionValue, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.fc1 = nn.Linear(obs_count + action_count, net_width)
        self.bn1 = BN(net_width)
        self.fc2 = nn.Linear(net_width, net_width // 2)
        self.bn2 = BN(net_width)
        self.fc3 = nn.Linear(net_width, net_width)
        self.bn3 = BN(net_width)
        self.fc4 = nn.Linear(net_width, net_width)
        self.bn4 = BN(net_width)
        self.fc5 = nn.Linear(net_width // 2, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat((s, a), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.tanh(self.bn3(self.fc3(x)))
        # x = torch.tanh(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

# State Value V network, input s, output expected total reward
class StateValue(nn.Module):
    def __init__(self, net_width, obs_count, usebn=False):
        super(StateValue, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.fc1 = nn.Linear(obs_count, net_width)
        self.bn1 = BN(net_width)
        self.fc2 = nn.Linear(net_width, net_width // 2)
        self.bn2 = BN(net_width // 2)
        self.fc3 = nn.Linear(net_width, net_width)
        self.bn3 = BN(net_width)
        self.fc4 = nn.Linear(net_width, net_width)
        self.bn4 = BN(net_width)
        self.fc5 = nn.Linear(net_width // 2, 1)

    def forward(self, s: torch.Tensor):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.tanh(self.bn3(self.fc3(x)))
        # x = torch.tanh(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        # x = torch.clamp(self.fc5(x), -1, 1)
        return x

def _update_target_net(targetnet, net, tau):
    new = collections.OrderedDict()
    a = targetnet.state_dict()
    b = net.state_dict()
    for k in a.keys():
        new[k] = a[k] * tau + b[k] * (1 - tau)
    targetnet.load_state_dict(new)

class TD3_Agent:
    def __init__(self, obs_count, action_count, net_width, stochastic=False, lr=1e-3, weight_decay=1e-3, usebn=False, device='cuda:0'):
        if stochastic:
            self.actor = Stochastic_Policy(net_width, obs_count, action_count, usebn=usebn).to(device)
            self.actor_target = Stochastic_Policy(net_width, obs_count, action_count, usebn=usebn).to(device)
        else:
            self.actor = Deterministic_Policy(net_width, obs_count, action_count, usebn=usebn).to(device)
            self.actor_target = Deterministic_Policy(net_width, obs_count, action_count, usebn=usebn).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=weight_decay)

        self.critic1 = ActionValue(net_width, obs_count, action_count, usebn=usebn).to(device)
        self.critic1_target = ActionValue(net_width, obs_count, action_count, usebn=usebn).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_target.eval()
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr, weight_decay=weight_decay)

        self.critic2 = ActionValue(net_width, obs_count, action_count, usebn=usebn).to(device)
        self.critic2_target = ActionValue(net_width, obs_count, action_count, usebn=usebn).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_target.eval()
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr, weight_decay=weight_decay)

        self.device = device
        self.train_steps = 0
    
    def load_checkpoint(self, actor_pth_path):
        self.actor.load_state_dict(torch.load(actor_pth_path, map_location=torch.device(self.device)))

    def act_stochastic(self, s, range=(-1, 1)):
        self.actor.eval()
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu, std = self.actor(s)
            # print(mu, std)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()
            torch.clamp(a, range[0], range[1])
            aprob = dist.log_prob(a)
            a = a.cpu().numpy()[0]
        self.actor.train()
        return a, aprob

    def act_deterministic(self, s):
        self.actor.eval()
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor(s)
            a = a.cpu().numpy()[0]
        self.actor.train()
        return a

    def train(self,
        batch_size: int,
        rb: ReplayBuffer,
        rb_steps: int,
        gamma: float,
        train_policy: bool = True,):
        if len(rb) < 20:
            return 0, 0
        self.train_steps += 1

        s, a, aprob, u, done_mask, s_next, a_next, asteps = rb.sample(batch_size, steps=rb_steps, device=self.device)
        with torch.no_grad():
            a_next = self.actor_target(s_next)
            # clipped noise regularization
            #policy_noise = 0.2
            #noise_clip = 0.5
            #noise = torch.zeros_like(a_next).normal_(0, policy_noise)
            #noise = noise.clamp(-noise_clip, noise_clip)
            #a_next = (a_next + noise).clamp(-1, 1)

            q1_target = u + (gamma ** asteps) * self.critic1_target(s_next, a_next) * done_mask
            q2_target = u + (gamma ** asteps) * self.critic2_target(s_next, a_next) * done_mask
            q_target = torch.min(q1_target, q2_target).detach()

        self.critic1_optimizer.zero_grad()
        self.critic1.requires_grad_(True)
        loss = F.smooth_l1_loss(self.critic1(s, a), q_target)
        ret1 = loss.item()
        loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        self.critic2.requires_grad_(True)
        loss = F.smooth_l1_loss(self.critic2(s, a), q_target)
        ret1 += loss.item()
        loss.backward()
        self.critic2_optimizer.step()

        ret2 = 0
        if train_policy and self.train_steps % 2 == 0:
            self.actor_optimizer.zero_grad()
            x = self.actor(s)
            self.critic1.requires_grad_(False)
            loss = -self.critic1(s, x).mean()
            ret2 = loss.item()
            loss.backward()
            self.actor_optimizer.step()
        return ret1, ret2

    def update_target(self, tau=0.995):
        _update_target_net(self.actor_target, self.actor, tau)
        _update_target_net(self.critic1_target, self.critic1, tau)
        _update_target_net(self.critic2_target, self.critic2, tau)
