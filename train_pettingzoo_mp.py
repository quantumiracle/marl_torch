import gym
import torch
torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work
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

import torch.multiprocessing as mp
from torch.multiprocessing import Process
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

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


def parallel_rollout(id, env_name, obs_type, model, max_eps, max_timesteps, render, seed):
    """ 
    Paralllel rollout for multi-agent games, in contrast to the iterative rollout manner.
    Parallel: (multi-agent actions are executed in once call of env.step())
    observations_, rewards, dones, infos = env.step(actions)
    actions, observations_, rewards, dones, infos are all dictionaries, 
    with agent name as key and corresponding values.
    """
    env = make_env(env_name, seed, obs_type=obs_type)
    env.reset() # required by env.agents
    score = {a:0.0 for a in env.agents}
    print_interval = 20
    save_interval = 100
    epi_len = []
    for n_epi in range(max_eps):
        observations = env.reset()

        for t in range(max_timesteps):
            actions, logprobs = model.choose_action(observations)
            observations_, rewards, dones, infos = env.step(actions)  # from discrete to multibinary action
            if render:
                env.render()
            
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
            record_score, record_length = {}, {}
            for agent_name in env.agents:
                avg_score = score[agent_name]/float(print_interval)
                avg_length = int(np.mean(epi_len))
                print("id : {}, agent :{}, avg score : {:.3f}, avg epi length : {}".format(id, agent_name, avg_score, avg_length))
                record_score[agent_name] = avg_score
                record_length[agent_name] = avg_length

            writer.add_scalars("ID {}/Scores".format(id, agent_name), record_score, n_epi)
            writer.add_scalars("ID {}/Episode Length".format(id, agent_name), record_length, n_epi)

            score = {a:0.0 for a in env.agents}
            epi_len = []
        if n_epi%save_interval==0 and n_epi!=0:
            model.save_model('model/mappo_mp')
    model.save_model('model/mappo_mp')

def ShareParameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()

def main():
    parser = argparse.ArgumentParser(description='Train or test arguments.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--env', type=str, help='Environment', required=True)
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
    if args.ram_obs or args.env == "slimevolley_v0":
        obs_type='ram'
    else:
        obs_type='rgb_image'
    env = make_env(args.env, SEED, obs_type=obs_type)
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
    if obs_type=='ram':
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'MLP', learner_args, **hyperparams).to(device)
    else:
        # model = PPODiscrete(state_space, action_space, 'CNN', learner_args, **hyperparams).to(device)
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'CNN', learner_args, **hyperparams).to(device)

    for individual_model in model.agents.values():
        individual_model.policy.share_memory()
        individual_model.policy_old.share_memory()
        individual_model.value.share_memory()
        ShareParameters(individual_model.optimizer)

    processes=[]
    for p in range(args.process):
        process = Process(target=parallel_rollout, args=(p, args.env, obs_type, model, \
            10000, max_timesteps, args.render, SEED))  # the args contain shared and not shared
        process.daemon=True  # all processes closed when the main stops
        processes.append(process)

    [p.start() for p in processes]

    [p.join() for p in processes]  # finished at the same time

    env.close()

if __name__ == '__main__':
    main()
