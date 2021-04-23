import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import numpy as np
import pettingzoo
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from utils.env import DummyVectorEnv, SubprocVectorEnv
from utils.arguments import get_args
from utils.wrappers import PettingZooWrapper, make_env
from utils.ppo import PPODiscrete, MultiPPODiscrete, ParallelMultiPPODiscrete
from utils.utils import create_log_dir, load_model
from hyperparams import *

def parallel_rollout(env, model, writer, max_eps, max_timesteps, selfplay_interval, render, \
    model_path, against_baseline=False, selfplay=False, fictitious=False, test=False):
    score = {a:0.0 for a in model.agents}
    print_interval =20
    save_interval = 100
    epi_len = []
    for n_epi in range(max_eps):
        observations = env.reset()
        for t in range(max_timesteps):
            actions, logprobs = model.choose_action(observations)
            # try:
            #     actions, logprobs = model.choose_action(observations)
            # except:
            #     print(t, observations)
            #     break
            if against_baseline:  # TODO there might be problem, against_baseline arg is not allowed when using VectorEnv
                observations_, rewards, dones, infos = env.step(actions, against_baseline)  # from discrete to multibinary action
            else:
                observations_, rewards, dones, infos = env.step(actions)
            if render:
                env.render()
            
            if not test:
                model.put_data((observations, actions, rewards, observations_, logprobs, dones))

            observations = observations_

            # If all envs with each having at least one agent is done, then finishe episode. (deprecated! does not work in this way!)
            # For example,
            # if dones= [{'first_0': True, 'second_0': True}, {'first_0': False, 'second_0': False}], it returns False;
            # if dones= [{'first_0': True, 'second_0': False}, {'first_0': True, 'second_0': False}], it returns True.
            # if np.all([np.any(np.array(list(d.values()))) for d in dones]):
            #     break

            # not env.agents: according to official docu (https://www.pettingzoo.ml/api), single agent will give {} if it recieved done, while others remain.
            # however, since env.agents is list of list, [[], []] will be bool True, but with one None filter it gives False.
            if not list(filter(None, env.agents)): 
                break 

            for agent_name in model.agents:
                rewards_=list(filter(None, rewards))  # filter out empty dicts caused by finished env episodes
                score[agent_name] += np.mean([r[agent_name] for r in rewards_]) # mean over different envs
        
        if not test:
            model.train_net()
            epi_len.append(t)
            # record training info
            if n_epi%print_interval==0 and n_epi!=0:
                print("# of episode :{}".format(n_epi))
                record_score, record_length = {}, {}
                for agent_name in model.agents:
                    avg_score = score[agent_name]/float(print_interval)
                    avg_length = int(np.mean(epi_len))
                    print("agent :{}, avg score : {:.3f}, avg epi length : {}".format(agent_name, avg_score, avg_length))
                    record_score[agent_name] = avg_score
                    record_length[agent_name] = avg_length

                writer.add_scalars("Scores".format(agent_name), record_score, n_epi)
                writer.add_scalars("Episode Length".format(agent_name), record_length, n_epi)
                
                score = {a:0.0 for a in model.agents}
                epi_len = []

            # selfplay load model
            if selfplay and n_epi%selfplay_interval==0 and n_epi!=0:
                save_model_path = model_path+'selfplay_mp/'+str(n_epi)+'mappo_single'
                model.save_model(save_model_path)
                print("Selfplay: update the model of opponent")
                # TODO different ways of opponent sampling in selfplay
                # 1. load the most recent one
                if not fictitious: 
                    load_model_path = save_model_path+'_1'
                # 2. load an average of historical model (ficticiou selfplay)
                else:  # fictitious selfplay
                    filelist=[]
                    for filename in os.listdir(model_path+'selfplay_mp/'):
                        if filename.endswith("policy"):
                            filelist.append('_'.join(filename.split('_')[:-1]))  # remove '_policy' at end
                    load_model_path = model_path+'selfplay_mp/' + filelist[np.random.randint(len(filelist))]
                model.load_model(agent_name='first_0', path=load_model_path)  # change the opponent


    #         if n_epi%save_interval==0 and n_epi!=0:
    #             model.save_model(model_path+'mappo_single_mp')
    # model.save_model(model_path+'mappo_single_mp')

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
    # env = make_env(args.env, SEED, obs_type=obs_type)
    VectorEnv = [DummyVectorEnv, SubprocVectorEnv][1]  # https://github.com/thu-ml/tianshou/blob/master/tianshou/env/venvs.py
    envs = VectorEnv([lambda: make_env(args.env, obs_type=obs_type) for _ in range(args.num_envs)])

    envs.seed(np.random.randint(1000, size=args.num_envs).tolist())  # random seeding
    
    state_spaces = envs.observation_spaces[0] # same for all env instances, so just take one
    action_spaces = envs.action_spaces[0] # same for all env instances, so just take one
    print('state_spaces: ', state_spaces, ',  action_spaces: ', action_spaces)

    learner_args = {'device':  args.device}
    envs.reset()
    agents = envs.agents[0] # same for all env instances, so just take one
    print('agents: ', agents)

    if args.train_both:
        fixed_agents = []
    else:
        fixed_agents = ['first_0']   # SlimeVolley: opponent is the first, the second agent is the learnable one

    if obs_type=='ram':
        model = ParallelMultiPPODiscrete(args.num_envs, agents, state_spaces, action_spaces, 'MLP', fixed_agents, learner_args, **hyperparams).to(args.device)
    else:
        model = ParallelMultiPPODiscrete(args.num_envs, agents, state_spaces, action_spaces, 'CNN', fixed_agents, learner_args, **hyperparams).to(args.device)
    
    load_model(model, args)

    path = f"model/{args.env}/"
    os.makedirs(path, exist_ok=True)
    
    if args.fictitious:
        path = path + 'fictitious_'

    parallel_rollout(envs, model, writer, max_eps=max_eps, max_timesteps=max_timesteps, selfplay_interval=selfplay_interval,\
        render=args.render, model_path=path, against_baseline=args.against_baseline, selfplay=args.selfplay, \
        fictitious=args.fictitious, test=args.test)

    envs.close()

if __name__ == '__main__':
    main()
