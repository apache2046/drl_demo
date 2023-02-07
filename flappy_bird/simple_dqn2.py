import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import pygame
import random
pygame.mixer.init()

import time
import flappy_bird_gym
import numpy as np
import wandb
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg("--play-with", type=str, default="", help="play game with specified saved model weight")
add_arg("--max-episodes", type=int, default=10000, help="")
add_arg("--gamma", type=float, default=0.98, help="")
add_arg("--batch-size", type=int, default=512, help="")
add_arg("--mstep", type=int, default=1, help="multiple step discount")
add_arg("--device", type=str, default="cuda:0", help="")

args = parser.parse_args()

learning_rate = 1e-4
gamma = args.gamma
device = args.device

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        # self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def select_action(self, state):
        with torch.no_grad():
            x = self.forward(state)
            # print(x)
            return x.argmax(-1)

class ReplayBuffer():
    def __init__(self, buffer_limit=100000):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def put_traj(self, traj):
        for transition in traj:
            self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_next_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_next, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_next_lst.append(s_next)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(device), \
               torch.tensor(a_lst).to(device), \
               torch.tensor(r_lst).to(device), \
               torch.tensor(s_next_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)

def train(rb:ReplayBuffer, qnet, qnet_target):
    losses = []
    for i in range(100):
        s, a, r, snext, done_mask = rb.sample(512)
        with torch.no_grad():
            a_next = qnet(snext).argmax(-1, keepdim=True)
            y_target = r + gamma* qnet_target(snext).gather(1, a_next) * done_mask
            if i == 0:
                print(y_target[:10].cpu().tolist())
        y = qnet(s).gather(1, a)
        loss = F.smooth_l1_loss(y, y_target)
        qnet.optimizer.zero_grad()
        loss.backward()
        qnet.optimizer.step()
        losses.append(loss.cpu().tolist())
    print("Loss:", np.mean(losses))

def update_target_net(targetnet, net, tau=0.1):
    new = collections.OrderedDict()
    a = targetnet.state_dict()
    b = net.state_dict()
    for k in a.keys():
        new[k] = a[k] * (1 - tau) + b[k] * tau
    targetnet.load_state_dict(new)

def play(pth_file):
    env = flappy_bird_gym.make("FlappyBird-v0")
    obs = env.reset()
    q = QNet()
    q = q.to(device)
    q.load_state_dict(torch.load(pth_file, map_location=torch.device(device)))
    while True:
        state = torch.tensor(obs, dtype=torch.float).to(device)
        action = q.select_action(state).cpu().tolist()
        # Processing:
        obs_new, reward, done, info = env.step(action)
        
        # Rendering the game:
        env.render()
        time.sleep(1 / 30)  # FPS
        if done:
            env.reset()


if args.play_with:
    play(args.play_with)


env = flappy_bird_gym.make("FlappyBird-v0")
wandb.init(project=os.path.basename(__file__).split(".")[0])

rb = ReplayBuffer()
q = QNet()
q_target = QNet()
q_target.load_state_dict(q.state_dict())
q = q.to(device)
q_target = q_target.to(device)

traj_cnt = 0
train_cnt = 0
random_ok_pushed = False
best_score = 4
while True:
    obs = env.reset()
    traj = []
    while True:
        # Next action:
        # (feed the observation to your agent here)
        if not random_ok_pushed:
            action = 0
            if random.random() > 29 / 30:
                action = 1
        else:
            state = torch.tensor(obs, dtype=torch.float).to(device)
            action = q.select_action(state).cpu().tolist()
        # Processing:
        obs_new, reward, done, info = env.step(action)
        done_mask = 0 if done else 1
        traj.append((obs, action, reward-action*0.3, obs_new, done_mask))

        obs = obs_new
        # print(action, obs, reward, done, info)
        
        # Checking if the player is still alive
        if done:
            if random_ok_pushed or info['score'] > 0:
                print(action, obs, reward, done, info, len(traj))
                if random_ok_pushed:
                    for i in traj:
                        print(i[1], end=" ")
                    print("")
                    wandb.log({'score:': info['score']})
                rb.put_traj(traj)
                traj_cnt +=1
                if info['score'] <= 2:
                    random_ok_pushed = not random_ok_pushed
                if info['score'] > best_score:
                    best_score = info['score']
                    torch.save(q.state_dict(), f"weights/{best_score}.pth")
            break
    if traj_cnt == 50:
        print("TRANING...")
        train(rb, q, q_target)
        traj_cnt = 0
        train_cnt += 1
    if train_cnt % 4 == 0:
        update_target_net(q_target, q)

