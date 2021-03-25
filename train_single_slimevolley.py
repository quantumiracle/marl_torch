import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import numpy as np
import slimevolleygym
from utils.ppo import PPODiscrete
from gym import spaces

# action transformation of SlimeVolley 
action_table = [[0, 0, 0], # NOOP
                [1, 0, 0], # LEFT (forward)
                [1, 0, 1], # UPLEFT (forward jump)
                [0, 0, 1], # UP (jump)
                [0, 1, 1], # UPRIGHT (backward jump)
                [0, 1, 0]] # RIGHT (backward)


def main():
    SEED = 721
    # env = gym.make('CartPole-v1')
    env = gym.make("SlimeVolley-v0")
    env.seed(SEED)
    state_space = env.observation_space
    action_dim = len(action_table)  # the action space of SlimeVolley is multibinary, which can be transformed from discrete
    action_space = spaces.Discrete(action_dim)
    print(state_space, action_space)

    hyperparams = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'hidden_dim': 64,
        'K_epoch': 4,
    }
    device_idx = 0
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    learner_args = {'device':  device}

    model = PPODiscrete(state_space, action_space, 'MLP', learner_args, **hyperparams).to(device)
    score = 0.0
    print_interval = 20
    epi_len = []
    for n_epi in range(10000):
        s = env.reset()
        done = False
        for t in range(env.t_limit):
            a, logprob = model.choose_action(s)
            s_prime, r, done, info = env.step(env.discreteToBox(a))  # from discrete to multibinary action
            # env.render()
            model.put_data((s, a, r, s_prime, logprob, done))

            s = s_prime

            score += r
            if done:
                break

        model.train_net()
        epi_len.append(t)
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.3f}, avg epi length : {}".format(n_epi, score/print_interval, int(np.mean(epi_len))))
            score = 0.0
            epi_len = []

    env.close()

if __name__ == '__main__':
    main()
