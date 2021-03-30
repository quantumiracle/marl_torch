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
import os
from hyperparams import *


# action transformation of SlimeVolley 
action_table = [[0, 0, 0], # NOOP
                [1, 0, 0], # LEFT (forward)
                [1, 0, 1], # UPLEFT (forward jump)
                [0, 0, 1], # UP (jump)
                [0, 1, 1], # UPRIGHT (backward jump)
                [0, 1, 0]] # RIGHT (backward)

def parallel_rollout(env, model, max_eps, max_timesteps, selfplay_interval, render, model_path, against_baseline=False):
    score = {a:0.0 for a in env.agents}
    avg_score = {a:[] for a in env.agents}
    epi_len = []
    for n_epi in range(max_eps):
        env.seed(np.random.randint(1000))  # take random seed for evaluation
        observations = env.reset()

        for t in range(max_timesteps):
            actions, logprobs = model.choose_action(observations)
            observations_, rewards, dones, infos = env.step(actions, against_baseline)  # from discrete to multibinary action
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
        for agent_name in env.agents:
            avg_score[agent_name].append(score[agent_name])
        score = {a:0.0 for a in env.agents}
        epi_len.append(t)

    avg_length = int(np.mean(epi_len))
    ag = 'second_0'
    print(ag+", avg score : {:.3f}, avg epi length : {}".format(np.mean(avg_score[ag]), avg_length))
    return np.mean(avg_score[ag]), avg_length

def main():
    parser = argparse.ArgumentParser(description='Train or test arguments.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--env', type=str, help='Environment', required=True)
    parser.add_argument('--ram', dest='ram_obs', action='store_true', default=False)
    parser.add_argument('--render', dest='render', action='store_true',
                    help='Enable openai gym real-time rendering')
    parser.add_argument('--seed', dest='seed', type=int, default=1234,
            help='Random seed')
    parser.add_argument('--load_agent', dest='load_agent', type=str, default=None, help='Load agent models by specifying: 1, 2, or both')
    parser.add_argument('--against_baseline', dest='against_baseline', action='store_true', default=False)
    args = parser.parse_args()

    SEED = np.random.randint(1000)
    if args.ram_obs or args.env == "slimevolley_v0":
        obs_type='ram'
    else:
        obs_type='rgb_image'
    env = make_env(args.env, SEED, obs_type=obs_type)
    # max_eps = 500000
    # max_timesteps = 10000
    # selfplay_interval = 3000 # interval in a unit of episode to checkpoint a policy and replace its opponent in selfplay
    eval_eps = 100

    state_spaces = env.observation_spaces
    action_spaces = env.action_spaces
    print('state_spaces: ', state_spaces, ',  action_spaces: ', action_spaces)

    device_idx = 0
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    learner_args = {'device':  device}
    env.reset()
    print(env.agents)
    agents = env.agents

    fixed_agents = ['first_0', 'second_0']   # SlimeVolley: opponent is the first, the second agent is the learnable one

    if obs_type=='ram':
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'MLP', fixed_agents, learner_args, **hyperparams).to(device)
    else:
        # model = PPODiscrete(state_space, action_space, 'CNN', learner_args, **hyperparams).to(device)
        model = MultiPPODiscrete(agents, state_spaces, action_spaces, 'CNN', fixed_agents, learner_args, **hyperparams).to(device)

    model_dir = 'model/selfplay/'
    filelist, epi_list = [], []
    for filename in os.listdir(model_dir):
        if filename.endswith("policy"):
            filelist.append('_'.join(filename.split('_')[:-1]))  # remove '_policy' at end
            epi_list.append(int(filename.split('mappo')[0]))
    epi_list.sort()
    filelist.sort()
    print(epi_list)

    r_list, l_list = [], []
    eval_data={}
    for f, i in zip(filelist, epi_list):
        print('episode: ', i)
        model.load_model(agent_name='second_0', path=model_dir+f)

        r, l = parallel_rollout(env, model, max_eps=eval_eps, max_timesteps=max_timesteps, selfplay_interval=selfplay_interval,\
            render=args.render, model_path=None, against_baseline=args.against_baseline)
        eval_data[str(i)]=[r, l]
    np.save('data/eval_data.npy', eval_data)

    env.close()

if __name__ == '__main__':
    main()
