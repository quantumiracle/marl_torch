import gym
import os
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
from utils.arguments import get_args
from utils.utils import create_log_dir, load_model
from hyperparams import *

def parallel_rollout(id, env_name, model, writer, max_eps, max_timesteps, selfplay_interval, render, \
    model_path, against_baseline=False, selfplay=False, fictitious=False, seed=0):
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

            writer.add_scalars("ID {}/Scores".format(id), record_score, n_epi)
            writer.add_scalars("ID {}/Episode Length".format(id), record_length, n_epi)

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
    args = get_args()
    log_dir = create_log_dir(args)
    if not args.test:
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    SEED = 721
    if args.ram_obs or args.env == "slimevolley_v0":
        obs_type='ram'
    else:
        obs_type='rgb_image'
    env = make_env(args.env, SEED, obs_type=obs_type)

    state_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    print('state_spaces: ', state_spaces, ',  action_spaces: ', action_spaces)

    learner_args = {'device':  args.device}
    env.reset()
    print(env.agents)
    agents = env.agents
    if args.train_both:
        fixed_agents = []
    else:
        fixed_agents = ['first_0']   # SlimeVolley: opponent is the first, the second agent is the learnable one

    if obs_type=='ram':
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'MLP', fixed_agents, learner_args, **hyperparams).to(args.device)
    else:
        # model = PPODiscrete(state_space, action_space, 'CNN', learner_args, **hyperparams).to(device)
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'CNN', fixed_agents, learner_args, **hyperparams).to(args.device)

    load_model(model, args)

    for individual_model in model.agents.values():
        individual_model.policy.share_memory()
        individual_model.policy_old.share_memory()
        individual_model.value.share_memory()
        ShareParameters(individual_model.optimizer)

    path = 'model/'+args.env
    os.makedirs(path, exist_ok=True)
    
    if args.fictitious:
        path = path + '/fictitious_'

    processes=[]
    for p in range(args.num_envs):
        process = Process(target=parallel_rollout, args=(p, args.env, model, writer, max_eps, \
            max_timesteps, selfplay_interval,\
            args.render, path, args.against_baseline, \
            args.selfplay, args.fictitious, SEED))  # the args contain shared and not shared
        process.daemon=True  # all processes closed when the main stops
        processes.append(process)

    [p.start() for p in processes]

    [p.join() for p in processes]  # finished at the same time

    env.close()

if __name__ == '__main__':
    main()
