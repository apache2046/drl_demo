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
from agentlib import TD3_Agent

sys.path.append(os.path.dirname(__file__) + "/..")

from utils import ReplayBuffer, OU_Noise

ENV_NAME = "Ant-v4"

learning_rate = 1e-3
tau = 0.9
device = "cuda:0"
weight_file = os.path.dirname(__file__) + "/weights/" + os.path.basename(__file__).split(".")[0] + "_3.pt"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg("--show", action="store_true", default=False, help="play game with last saved model weight instead of train")
add_arg("--continuous-actions", type=int, default=0, help="")
add_arg("--max-episodes", type=int, default=100000, help="")
add_arg("--ou-max-episodes", type=int, default=5000, help="")
add_arg("--gamma", type=float, default=0.98, help="")
add_arg("--eps_clip", type=float, default=0.1, help="")
add_arg("--batch-size", type=int, default=512, help="")
add_arg("--mstep", type=int, default=1, help="multiple step discount")
add_arg("--trajectories-limit", type=int, default=4000, help="")

args = parser.parse_args()

def show(agent: TD3_Agent, pth_file):
    agent.load_checkpoint(pth_file)
    env = gym.make(ENV_NAME, ctrl_cost_weight=1e-3,  render_mode='human', healthy_z_range=(0.3, 1))
    # env.unwrapped.frame_skip = 20
    while True:
        s, _ = env.reset()
        done = False
        while not done:
            env.render()
            a = agent.act_deterministic(s)
            s, r, done, truncated, info = env.step(a)
            if done or truncated:
                break

def main():
    wandb.init(project=os.path.basename(__file__).split(".")[0])
    config = wandb.config
    config.ver = 1
    config.update(args)
    batch_size = args.batch_size

    env = gym.make(ENV_NAME)#, ctrl_cost_weight=1e-3)
    # env.unwrapped.frame_skip = 20
    action_count = env.action_space.shape[0]
    observation_count = env.observation_space.shape[0]

    agent = TD3_Agent(observation_count, action_count, batch_size, stochastic=False, lr=1e-3, weight_decay=1e-3, usebn=False, device='cuda:0')
    
    if True:
        show(agent, "weights/ant2.pt")

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
                if n_epi < 20:
                    action = env.action_space.sample()
                else:
                    action = agent.act_deterministic(s)
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
                    r += _r #info['reward_forward']
                    steps += 1
                    if done:
                        break
                # if steps >= 10000:  # max(300, 2000 * min(1, n_epi / 30000)):
                #     done = True
                if done:
                    done_mask = 0.0
                else:
                    done_mask = 1.0
                r1 = r
                # if r <= -100:
                #     r1 = -5
                rb.put_transition((s.tolist(), action, 0, r1, done_mask))
                s = s_prime
                total_reward += r
            total_steps += steps

        # print(n_epi, f"{total_reward / play_cnt :.1f}, {total_steps / play_cnt:.1f}")
        # total_reward = 0
        # total_steps = 0
        for i in range(200):
            l1, l2 = agent.train(512, rb, args.mstep, gamma=args.gamma, train_policy=True)
            losses1.append(l1)
            losses2.append(l2)
            if i % 2 == 0:
                agent.update_target(tau=0.995)

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
                torch.save(agent.actor.state_dict(), weight_file)
                print("saved a pt file...")
                # args.continuous_actions = max(0, args.continuous_actions - 1)
            total_steps = 0.0
            total_reward = 0.0
            losses1 = []
            losses2 = []

    env.close()

if __name__ == "__main__":
    main()
