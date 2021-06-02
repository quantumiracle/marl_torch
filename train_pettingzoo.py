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

from utils.wrappers import PettingZooWrapper, make_env
from utils.ppo import PPODiscrete, MultiPPODiscrete
from utils.arguments import get_args
from utils.utils import create_log_dir, load_model
from hyperparams import *

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


def parallel_rollout(env, model, writer, max_eps, max_timesteps, selfplay_interval, render, \
    model_path, against_baseline=False, selfplay=False, fictitious=False, test=False):
    score = {a:0.0 for a in env.agents}
    print_interval =20
    save_interval = 100
    epi_len = []
    for n_epi in range(max_eps):
        observations = env.reset()

        for t in range(max_timesteps):
            # noise = np.random.normal(0, 0.1, size=observations['first_0'].shape[0])
            # observations['first_0'] = observations['first_0'] + noise
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

            for agent_name in env.agents:
                score[agent_name] += rewards[agent_name]

            if np.any(np.array(list(dones.values()))):  # any agent has a done -> terminate episode
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
                for agent_name in env.agents:
                    avg_score = score[agent_name]/float(print_interval)
                    avg_length = int(np.mean(epi_len))
                    print("agent :{}, avg score : {:.3f}, avg epi length : {}".format(agent_name, avg_score, avg_length))
                    record_score[agent_name] = avg_score
                    record_length[agent_name] = avg_length
                writer.add_scalars("Scores", record_score, n_epi)
                writer.add_scalars("Episode Length", record_length, n_epi)
            
                score = {a:0.0 for a in env.agents}
                epi_len = []

            # selfplay load model
            if selfplay and n_epi%selfplay_interval==0 and n_epi!=0:
                save_model_path = model_path+'selfplay/'+str(n_epi)+'mappo_single'
                model.save_model(save_model_path)
                print("Selfplay: update the model of opponent")
                # TODO different ways of opponent sampling in selfplay
                # 1. load the most recent one
                if not fictitious: 
                    load_model_path = save_model_path+'_1'
                # 2. load an average of historical model (ficticiou selfplay)
                else:  # fictitious selfplay (not standard)
                    filelist=[]
                    for filename in os.listdir(model_path+'selfplay/'):
                        if filename.endswith("policy"):
                            filelist.append('_'.join(filename.split('_')[:-1]))  # remove '_policy' at end
                    load_model_path = model_path+'selfplay/' + filelist[np.random.randint(len(filelist))]
                model.load_model(agent_name='first_0', path=load_model_path)  # change the opponent


            if n_epi%save_interval==0 and n_epi!=0:
                model.save_model(model_path+'mappo_single')
    model.save_model(model_path+'mappo_single')

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
    path = f"model/{args.env}/"
    os.makedirs(path, exist_ok=True)
    data_path = f"data/{args.env}/"
    os.makedirs(data_path, exist_ok=True)
    
    if obs_type=='ram':
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'MLP', fixed_agents, learner_args, **hyperparams).to(args.device)
    else:
        # model = PPODiscrete(state_space, action_space, 'CNN', learner_args, **hyperparams).to(device)
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'CNN', fixed_agents, learner_args, **hyperparams).to(args.device)
        path = path + 'cnn_'

    load_model(model, args)

    if args.fictitious:
        path = path + 'fictitious_'

    parallel_rollout(env, model, writer, max_eps=max_eps, max_timesteps=max_timesteps, selfplay_interval=selfplay_interval,\
        render=args.render, model_path=path, against_baseline=args.against_baseline, selfplay=args.selfplay, \
        fictitious=args.fictitious, test=args.test)

    env.close()

if __name__ == '__main__':
    main()
