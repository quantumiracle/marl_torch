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
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from utils.wrappers import PettingZooWrapper, make_env
from utils.ppo import PPODiscrete, MultiPPODiscrete, ParallelMultiPPODiscrete
from utils.arguments import get_args
from utils.utils import create_log_dir, load_model

# action transformation of SlimeVolley 
action_table = [[0, 0, 0], # NOOP
                [1, 0, 0], # LEFT (forward)
                [1, 0, 1], # UPLEFT (forward jump)
                [0, 0, 1], # UP (jump)
                [0, 1, 1], # UPRIGHT (backward jump)
                [0, 1, 0]] # RIGHT (backward)

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
            if against_baseline:
                observations_, rewards, dones, infos = env.step(actions, against_baseline)  # from discrete to multibinary action
            else:
                observations_, rewards, dones, infos = env.step(actions)
            if render:
                env.render()
            
            if not test:
                model.put_data((observations, actions, rewards, observations_, logprobs, dones))

            observations = observations_

            for agent_name in model.agents:
                score[agent_name] += np.mean([r[agent_name] for r in rewards]) # mean over different envs

            if np.all([np.any(np.array(list(d.values()))) for d in dones]):
                # If all envs with each having at least one agent is done, then finishe episode.
                # For example,
                # if dones= [{'first_0': True, 'second_0': True}, {'first_0': False, 'second_0': False}], it returns False;
                # if dones= [{'first_0': True, 'second_0': False}, {'first_0': True, 'second_0': False}], it returns True.
                break

            # if not env.agents: # according to official docu (https://www.pettingzoo.ml/api), single agent will be removed if it recieved done, while others remain 
            #     break 

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


            if n_epi%save_interval==0 and n_epi!=0:
                model.save_model(model_path+'mappo_single_mp')
    model.save_model(model_path+'mappo_single_mp')

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
    env = make_env(args.env, SEED, obs_type=obs_type)  # TODO used for providing spaces info, can also modify SubprocVecEnv wrapper
    # https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html?highlight=multiprocessing
    envs = SubprocVecEnv([lambda: make_env(args.env, obs_type=obs_type) for _ in range(args.num_envs)], start_method='spawn')

    # envs.seed(np.random.randint(1000, size=args.num_envs).tolist())  # random seeding
    envs.seed(SEED)  # fix seeding
    
    max_eps = 500000
    max_timesteps = 10000
    selfplay_interval = 3000 # interval in a unit of episode to checkpoint a policy and replace its opponent in selfplay

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
    learner_args = {'device':  args.device}
    env.reset()
    agents = env.agents
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
