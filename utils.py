import torch
import random
import collections
import json
import numpy as np
import copy

class ReplayBuffer:
    def __init__(self, maxlen, gamma=1.0):
        self.trajectories = collections.deque(maxlen=maxlen)
        self.gamma = gamma

    def new_trajectory(self):
        self.trajectories.append([])

    def delete_trajectory(self):
        self.trajectories.pop()

    def put_transition(self, transition):
        self.trajectories[-1].append(transition)

    def sample(self, n, steps=1, device="cpu"):
        s_arr, a_arr, aprob_arr, u_arr, done_mask_arr, s_next_arr, a_next_arr, asteps_arr = [], [], [], [], [], [], [], []

        for _ in range(n):
            trajectory = random.sample(self.trajectories, 1)[0]
            traj_len = len(trajectory)
            trans_idx = random.choice(range(traj_len))
            asteps = min(steps, traj_len - trans_idx)
            s, a, aprob, _, _ = trajectory[trans_idx]
            u = 0
            for i in range(asteps):
                _, _, _, r, done_mask = trajectory[trans_idx + i]
                u += (self.gamma ** i) * r

            if done_mask == 0:
                s_next = s
                a_next = a
            else:
                s_next, a_next, _, _, _ = trajectory[trans_idx + asteps]

            s_arr.append(s)
            a_arr.append(a)
            aprob_arr.append(aprob)
            u_arr.append(u)
            asteps_arr.append(asteps)
            done_mask_arr.append(done_mask)
            s_next_arr.append(s_next)
            a_next_arr.append(a_next)

        return (
            torch.tensor(np.array(s_arr)).float().to(device),
            torch.tensor(np.array(a_arr)).float().to(device),
            torch.tensor(np.array(aprob_arr)).float().to(device),
            torch.tensor(np.array(u_arr)).float().unsqueeze(1).to(device),
            torch.tensor(np.array(done_mask_arr)).unsqueeze(1).float().to(device),
            torch.tensor(np.array(s_next_arr)).float().to(device),
            torch.tensor(np.array(a_next_arr)).float().to(device),
            torch.tensor(np.array(asteps_arr)).float().unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.trajectories)


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config, defaults):
    if type(str) == str:
        with open(config, "r") as fd:
            config = json.load(fd)
    _merge(defaults, config)
    return config

def update_target_net(targetnet, net, t):
    new = collections.OrderedDict()
    a = targetnet.state_dict()
    b = net.state_dict()
    for k in a.keys():
        new[k] = a[k] * t + b[k] * (1 - t)
    targetnet.load_state_dict(new)

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state
