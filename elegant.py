import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import argparse
from algorithms import DQN, DQN_train

def main(args) :
    env = gym.make('CartPole-v0')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn = DQN(args, state_space, action_space)

    dqn.cartpole_train_loop(env)


if __name__ == '__main__' :
    parse = argparse.ArgumentParser()

    parse.add_argument('--batch_size', default=256)
    parse.add_argument('--lr', default=0.003)
    parse.add_argument('--epsilon_decay', default=0.99)
    parse.add_argument('--gamma', default=0.99)
    parse.add_argument('--target_replace_iter', default=100)
    parse.add_argument('--memory_capacity', default=1000)
    parse.add_argument('--num_episode', default=1000)
    parse.add_argument('--ep_print_iter', default=10)
    parse.add_argument('--model_save_iter', default=100)

    args = parse.parse_args()

    main(args)