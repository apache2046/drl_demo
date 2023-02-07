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

sys.path.append(os.path.dirname(__file__) + "/..")

from utils import ReplayBuffer, OU_Noise

ENV_NAME = "Ant-v4"

learning_rate = 1e-3
tau = 0.9
device = "cuda:0"
weight_file = os.path.dirname(__file__) + "/weights/" + os.path.basename(__file__).split(".")[0] + ".pt"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg("--show", action="store_true", default=False, help="play game with last saved model weight instead of train")
add_arg("--continuous-actions", type=int, default=0, help="")
add_arg("--max-episodes", type=int, default=100000, help="")
add_arg("--ou-max-episodes", type=int, default=5000, help="")
add_arg("--gamma", type=float, default=0.98, help="")
add_arg("--eps_clip", type=float, default=0.2, help="")
add_arg("--batch-size", type=int, default=512, help="")
add_arg("--mstep", type=int, default=1, help="multiple step discount")
add_arg("--trajectories-limit", type=int, default=100, help="")

args = parser.parse_args()

NET_W = 512
# Deterministic policy network, input s, output the action with n degrees of freedom
class Policy(nn.Module):
    def __init__(self, obs_count, action_count, usebn=False):
        super(Policy, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.fc1 = nn.Linear(obs_count, NET_W)
        self.bn1 = BN(NET_W)
        self.fc2 = nn.Linear(NET_W, NET_W // 2)
        self.bn2 = BN(NET_W // 2)
        self.fc3 = nn.Linear(NET_W, NET_W)
        self.bn3 = BN(NET_W)
        self.fc4 = nn.Linear(NET_W, NET_W)
        self.bn4 = BN(NET_W)
        self.fc5 = nn.Linear(NET_W // 2, action_count)
        self.fc6 = nn.Linear(NET_W // 2, action_count)
        self.std = nn.Parameter(torch.zeros((1, action_count)), requires_grad=True)  # trainable parameter
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-3)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, s: torch.Tensor):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.relu(self.bn3(self.fc3(x)))
        # x = torch.relu(self.bn4(self.fc4(x)))
        mu = torch.tanh(self.fc5(x))
        # std = torch.sigmoid(self.fc6(x))
        std = torch.exp(self.std)
        # x = torch.clamp(self.fc5(x), -1, 1)
        # print(mu.shape, std.shape)
        return mu, std




# Action Value Q network, input s and a, output the expected total reward
class ActionValue(nn.Module):
    def __init__(self, obs_count, action_count, usebn=False):
        super(ActionValue, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.fc1 = nn.Linear(obs_count + action_count, NET_W)
        self.bn1 = BN(NET_W)
        self.fc2 = nn.Linear(NET_W, NET_W // 2)
        self.bn2 = BN(NET_W // 2)
        self.fc3 = nn.Linear(NET_W, NET_W)
        self.bn3 = BN(NET_W)
        self.fc4 = nn.Linear(NET_W, NET_W)
        self.bn4 = BN(NET_W)
        self.fc5 = nn.Linear(NET_W // 2, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-3)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat((s, a), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.relu(self.bn3(self.fc3(x)))
        # x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

# State Value V network, input s, output expected total reward
class StateValue(nn.Module):
    def __init__(self, obs_count, usebn=False):
        super(StateValue, self).__init__()
        BN = nn.BatchNorm1d if usebn else nn.Identity
        self.fc1 = nn.Linear(obs_count, NET_W)
        self.bn1 = BN(NET_W)
        self.fc2 = nn.Linear(NET_W, NET_W // 2)
        self.bn2 = BN(NET_W // 2)
        self.fc3 = nn.Linear(NET_W, NET_W)
        self.bn3 = BN(NET_W)
        self.fc4 = nn.Linear(NET_W, NET_W)
        self.bn4 = BN(NET_W)
        self.fc5 = nn.Linear(NET_W // 2, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, s: torch.Tensor):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.relu(self.bn3(self.fc3(x)))
        # x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        # x = torch.clamp(self.fc5(x), -1, 1)
        return x

def update_target_net(targetnet, net, t=tau):
    new = collections.OrderedDict()
    a = targetnet.state_dict()
    b = net.state_dict()
    for k in a.keys():
        new[k] = a[k] * t + b[k] * (1 - t)
    targetnet.load_state_dict(new)

def train(
    actor: Policy,
    actor_target: Policy,
    critic1: StateValue,
    critic1_target: StateValue,
    rb: ReplayBuffer,
    rb_steps,
    train_policy: bool,
):
    if len(rb) < 20:
        return 0, 0
    train.steps += 1
    actor.train()
    critic1.train()
    s, a, aprob, u, done_mask, s_next, a_next, asteps = rb.sample(args.batch_size, steps=rb_steps, device=device)
    with torch.no_grad():
        # a_next = actor_target(s_next)
        # clipped noise regularization
        #policy_noise = 0.2
        #noise_clip = 0.5
        #noise = torch.zeros_like(a_next).normal_(0, policy_noise)
        #noise = noise.clamp(-noise_clip, noise_clip)
        #a_next = (a_next + noise).clamp(-1, 1)

        # v1_target = u + (args.gamma ** asteps) * critic1_target(s_next) * done_mask
        v1_target = u + (args.gamma ** asteps) * critic1(s_next) * done_mask
        # q2_target = u + (args.gamma ** asteps) * critic2_target(s_next, a_next) * done_mask
        # q_target = torch.min(q1_target, q2_target).detach()

    critic1.optimizer.zero_grad()
    critic1.requires_grad_(True)
    loss = F.smooth_l1_loss(critic1(s), v1_target)
    ret1 = loss.item()
    loss.backward()
    nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=0.5)
    critic1.optimizer.step()

    ret2 = 0
    if train_policy and train.steps % 3 == 0:
        mu, std = actor(s)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)
        prob = log_prob.sum(dim=1)
        # print("prob.shape", prob.shape)
        old_prob = aprob.sum(dim=1)
        ratio = torch.exp(prob - old_prob)
        critic1.requires_grad_(False)
        advantage = v1_target - critic1(s)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - args.eps_clip, 1 + args.eps_clip) * advantage
        loss = -torch.min(surr1, surr2).mean()

        actor.optimizer.zero_grad()
        ret2 = loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        actor.optimizer.step()
    return ret1, ret2


train.steps = 0


def show():
    env = gym.make(ENV_NAME, render_mode='human')
    pi = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    pi.load_state_dict(torch.load(weight_file, map_location=torch.device(device)))
    pi.to(device)
    for runs in range(10):
        s, _ = env.reset()
        env.render()
        steps = 0
        total_steps = 0
        action = 0
        while True:
            action = select_action(pi, s)
            for _ in range(args.continuous_actions):
                env.step(action)[1]
            s, r, done, truncated, info = env.step(action)
            done = done or truncated
            env.render()
            steps += 1
            if done:
                break
            total_steps += r
        print(f"#{runs} done, steps:{steps}, total_steps:{total_steps}")
    env.close()


def select_action(policy: Policy, s, n_epi, ou_noise):
    with torch.no_grad():
        s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        # print(s.shape)
        policy.eval()
        mu, std = policy(s)
        mu = mu[0]
        std = std[0]
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        #action += torch.tensor(ou_noise.sample() * max(0, 1 - n_epi / 1000)).to(device)
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy()

def main():
    if args.show:
        return show()
    wandb.init(project=os.path.basename(__file__).split(".")[0])
    config = wandb.config
    config.ver = 1
    config.update(args)

    env = gym.make(ENV_NAME, ctrl_cost_weight=1e-3)
    env.unwrapped.frame_skip = 20
    action_count = env.action_space.shape[0]
    observation_count = env.observation_space.shape[0]
    pi = Policy(observation_count, action_count)
    vnet = StateValue(observation_count)

    torch.nn.init.constant_(pi.std, -0.5)
    # for m in list(pi.modules()) + list(vnet.modules()):
    #     if isinstance(m, torch.nn.Linear):
    #         # orthogonal initialization
    #         torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
    #         torch.nn.init.zeros_(m.bias)
    # # do last policy layer scaling, this will make initial actions have (close to)
    # # 0 mean and std, and will help boost performances,
    # # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    # for m in pi.mu.modules():
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.zeros_(m.bias)
    #         m.weight.data.copy_(0.01 * m.weight.data)


    pi_target = Policy(observation_count, action_count)
    pi_target.load_state_dict(pi.state_dict())

    vnet_target = StateValue(observation_count)
    vnet_target.load_state_dict(vnet.state_dict())

    pi.to(device)
    pi_target.to(device)
    vnet.to(device)
    vnet_target.to(device)


    # ou noise
    ou_noise = OU_Noise(size=action_count, seed=1, mu=0, theta=0.5, sigma=0.05)
    ou_noise.reset()

    total_steps = 0.0
    total_reward = 0.0
    print_interval = 1
    max_avg_reward = 50
    losses1 = []
    losses2 = []
    rb = ReplayBuffer(gamma=args.gamma, maxlen=args.trajectories_limit)

    noise_ratio = 0.35
    play_cnt = 100
    for n_epi in range(1, args.max_episodes):
        noise_ratio *= 0.999
        for i in range(play_cnt):
            s, _ = env.reset()
            need_render = True if i == 0 and n_epi % 10 == 0 else False

            done = False

            rb.new_trajectory()
            steps = 0
            action = 0
            while not done:
                action, action_prob = select_action(pi, s, n_epi, ou_noise)
                # shift_action = np.random.normal(0, 0.1, size=4) * noise_ratio
                # action = (action + shift_action).clip(-1, 1)
                # action += ou_noise.sample() * max(0, 1 - n_epi / 200)
                # tt = max(0, 1 - n_epi / args.ou_max_episodes)
                # action = (1 - tt) * action + tt * ou_noise.sample()
                # action = np.clip(action, -1, 1)
                r = 0
                for _ in range(1):  # range(min(1, 10 - n_epi // 100)):
                    s_prime, _r, done, truncated, info = env.step(action)
                    done = done or truncated
                    # if need_render:
                    #     env.render()
                    # r += _r # - 0.1
                    r += info['reward_forward']
                    steps += 1
                    if done:
                        break
                # if steps >= 1600:  # max(300, 2000 * min(1, n_epi / 30000)):
                #     done = True
                if done:
                    done_mask = 0.0
                else:
                    done_mask = 1.0
                r1 = r
                # if r <= -100:
                #     r1 = -5
                rb.put_transition((s.tolist(), action, action_prob, r1, done_mask))
                s = s_prime
                total_reward += r
            total_steps += steps

        # print(n_epi, f"{total_reward / play_cnt :.1f}, {total_steps / play_cnt:.1f}")
        # total_reward = 0
        # total_steps = 0
        for _ in range(100):
            l1, l2 = train(pi, pi_target, vnet, vnet_target, rb, args.mstep, train_policy=True)
            losses1.append(l1)
            losses2.append(l2)
            # update_target_net(pi_target, pi, 0.995)
            # update_target_net(vnet_target, vnet, 0.995)

        if n_epi % print_interval == 0:
            total_steps /= play_cnt
            total_reward /= play_cnt
            print(
                f"# of episode :{n_epi:04d}, "
                f"avg steps : {total_steps / print_interval:05.2f}, "
                f"avg rewrd : {total_reward / print_interval:05.2f}, "
                f"avg loss1:{np.array(losses1).mean():.8f}, "
                f"avg loss2:{np.array(losses2).mean():.8f}, "
                , info
            )
            wandb.log(
                {
                    "episode": n_epi,
                    "avg_steps": total_steps / print_interval,
                    "avg_rewards": total_reward / print_interval,
                }
            )
            if total_reward / print_interval > max_avg_reward:
                max_avg_reward = total_reward / print_interval
                # torch.save(pi.state_dict(), weight_file)
                print("saved a pt file...")
                # args.continuous_actions = max(0, args.continuous_actions - 1)
            total_steps = 0.0
            total_reward = 0.0
            losses1 = []
            losses2 = []

    env.close()

if __name__ == "__main__":
    main()
