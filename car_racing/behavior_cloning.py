import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import argparse
import torch.nn.functional as F
import numpy as np
import os
import sys
import wandb
import time
import torchvision
import pickle
import random

sys.path.append(os.path.dirname(__file__) + "/..")

from utils import ReplayBuffer, OU_Noise

ENV_NAME = "CarRacing-v2"

learning_rate = 0.0001
tau = 0.9
device = "cuda:0"
weight_file_prefix = os.path.dirname(__file__) + "/bc_weights/" + os.path.basename(__file__).split(".")[0]
batch_size = 512
sample_data = []

NET_W = 512
# Deterministic policy network, input s, output the action with n degrees of freedom
class Policy(nn.Module):
    def __init__(self, action_count, usebn=False):
        super(Policy, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.cnn = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        # Freeze the second linear layer
        # for param in self.cnn.parameters():
        #     param.requires_grad = False
        self.cnn.classifier = nn.Linear(1280, NET_W)
        self.bn1 = BN(NET_W)

        self.fc2 = nn.Linear(NET_W, NET_W // 2)
        self.bn2 = BN(NET_W // 2)
        self.fc3 = nn.Linear(NET_W // 2, action_count)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-3)

    def forward(self, s: torch.Tensor):
        x = torch.relu(self.bn1(self.cnn(s)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x

def get_sample():
    samples = random.sample(sample_data, batch_size)
    s_arr = []
    action_arr = []
    for item in samples:
       s, action, r, done = item
       s_arr.append(s)
       action_arr.append(action)

    return (torch.tensor(np.array(s_arr)).float().to(device),
            torch.tensor(np.array(action_arr)).float().to(device))

# loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.L1Loss()
loss_fn = torch.nn.SmoothL1Loss(beta=0.4)

def train(pi):
    pi.train()
    loss_arr = []
    for _ in range(100):
        s, a = get_sample()
        s = s.permute(0, 3, 1, 2)
        pred = pi(s)
        loss = loss_fn(pred, a)
        pi.optimizer.zero_grad()
        loss.backward()
        pi.optimizer.step()
        loss_arr.append(loss.detach().cpu().numpy())
    return np.array(loss_arr).mean()



for i in range(0, 210):
    with open(f'tracks/{i:04d}.pickle', "rb") as f:
        data = pickle.load(f)
        for s, action, r, done, truncated in data:
            # print(action)
            s = (np.array(s) - 128) / 255
        # for s, r, done in data:
            # action = np.random.randn(3).clip(-0.9, 0.9)
            steer, throttle, brake = action
            if throttle > brake:
                accel = throttle
            elif throttle < brake:
                accel = -brake
            else:
                accel = 0
            sample_data.append((s, [steer, accel], r, done))


pi = Policy(action_count = 2, usebn = False)
pi = pi.to(device)

ENV_NAME = "CarRacing-v2"
env = gym.make(ENV_NAME, render_mode='human')
def eval_play():
    s, _ = env.reset()
    done = False
    truncated = False
    pi.eval()
    while not (done or truncated):
        s = (torch.tensor(s).permute(2, 0, 1) - 128)/ 255
        s = s.unsqueeze(0).float().to(device)
        # print(s.shape)
        with torch.no_grad():
            action = pi(s).detach().cpu().numpy()[0]
        steer = action[0]
        if action[1] > 0:
            throttle = action[1]
            brake = 0
        else:
            throttle = 0
            brake = -action[1]
        print([steer, throttle, brake])
        sp, r, done, truncated, info = env.step([steer, throttle, brake])
        s = sp
    env.render()
for it in range(1, 100):
    l = train(pi)
    print("loss:", l)
    if it % 5 == 0:
        for _ in range(3):
            eval_play()
        torch.save(pi.state_dict(), weight_file_prefix + f'_{it}_{l:.1f}.pt' )

