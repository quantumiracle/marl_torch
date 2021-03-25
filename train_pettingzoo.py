import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import numpy as np
import pettingzoo
from utils.wrappers import PettingZooWrapper, make_env
from utils.ppo import PPODiscrete, MultiPPODiscrete
import argparse

# action transformation of SlimeVolley 
action_table = [[0, 0, 0], # NOOP
                [1, 0, 0], # LEFT (forward)
                [1, 0, 1], # UPLEFT (forward jump)
                [0, 0, 1], # UP (jump)
                [0, 1, 1], # UPRIGHT (backward jump)
                [0, 1, 0]] # RIGHT (backward)

def iterate_rollout(env, model, max_eps, max_timesteps):
    """ TODO the PettingZooWrapper also needs to be modified for this usage"""
    score = 0.0
    print_interval = 2
    epi_len = []
    for n_epi in range(max_eps):
        s = env.reset()
        done = False
        t = 0
        for agent in env.agent_iter():
            t+=1

            obs, r, done, info = env.last()
            a, logprob = model.choose_action(obs)
            env.step(a)
            env.render()
            score += r

            if done:
                break

            # TODO put data correctly in buffer: separate or not? with next obs or not
                model.put_data((s, a, r, s_prime, logprob, done))

        model.train_net()
        epi_len.append(t)
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.3f}, avg epi length : {}".format(n_epi, score/print_interval, int(np.mean(epi_len))))
            score = 0.0
            epi_len = []
            model.save_model('model/mappo')


def parallel_rollout(env, model, max_eps, max_timesteps):
    score = {a:0.0 for a in env.agents}
    print_interval = 20
    epi_len = []
    for n_epi in range(max_eps):
        observations = env.reset()

        for t in range(max_timesteps):
            actions, logprobs = model.choose_action(observations)
            observations_, rewards, dones, infos = env.step(actions)  # from discrete to multibinary action
            # env.render()
            
            model.put_data((observations, actions, rewards, observations_, logprobs, dones))

            observations = observations_

            for agent_name in env.agents:
                score[agent_name] += rewards[agent_name]

            if np.any(np.array(list(dones.values()))):  # any agent has a done -> terminate episode
                break

            # if not env.agents: # according to official docu (https://www.pettingzoo.ml/api), single agent will be removed if it recieved done, while others remain 
            #     break 

        model.train_net()
        epi_len.append(t)
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}".format(n_epi))
            for agent_name in env.agents:
                print("agent :{}, avg score : {:.3f}, avg epi length : {}".format(agent_name, score[agent_name]/print_interval, int(np.mean(epi_len))))
            score = {a:0.0 for a in env.agents}
            epi_len = []
            model.save_model('model/mappo')

def main():
    parser = argparse.ArgumentParser(description='Train or test arguments.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--env', type=str, help='Environment', required=False)
    parser.add_argument('--ram', dest='ram_obs', action='store_true', default=False)
    parser.add_argument('--render', dest='render', action='store_true',
                    help='Enable openai gym real-time rendering')
    parser.add_argument('--process', type=int, default=1,
                    help='Process count for parallel exploration')
    parser.add_argument('--seed', dest='seed', type=int, default=1234,
            help='Random seed')
    parser.add_argument('--alg', dest='alg', type=str, default='td3',
                help='Choose algorithm type')
    args = parser.parse_args()

    SEED = 721
    if args.ram_obs:
        env = make_env('boxing_v1', SEED, obs_type='ram')
    else:
        env = make_env('boxing_v1', SEED, obs_type='rgb_image')
    max_timesteps = 10000
    state_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    print('state_spaces: ', state_spaces, ',  action_spaces: ', action_spaces)
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
    env.reset()
    print(env.agents)
    agents = env.agents
    if args.ram_obs:
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'MLP', learner_args, **hyperparams).to(device)
    else:
        # model = PPODiscrete(state_space, action_space, 'CNN', learner_args, **hyperparams).to(device)
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'CNN', learner_args, **hyperparams).to(device)

    parallel_rollout(env, model, max_eps=10000, max_timesteps=max_timesteps) 
    env.close()

if __name__ == '__main__':
    main()
