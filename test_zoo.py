"""
Testing script for PettingZoo environments.
PettingZoo as a multi-agent library provides two environment interaction modes:
1. iterate: the action of each agent is taken iteratively in order with env.step(ai) function;
2. parallel: the actions of all agents are taken at the same env.step(a1, a2, ...) function.
"""

# from pettingzoo.butterfly import knights_archers_zombies_v7
# env = knights_archers_zombies_v7.env()
from pettingzoo.atari import boxing_v1, pong_v1
from pettingzoo.classic import rps_v1, rpsls_v1, leduc_holdem_v3, texas_holdem_v3, tictactoe_v3
from utils.wrappers import PettingZooWrapper
import gym
import supersuit
import argparse
import random
import numpy as np


def run_iterate(args, policy=None):
    """
    env.step(a) function iteratively takes the action of each agent.
    """
    env = eval(args.env).env()
    for _ in range(3):
        env.reset()
        step = 0
        for agent in env.agent_iter():
            step+=1
            print(step)
            observation, reward, done, info = env.last()
            print(observation, reward, done, info)
            if policy is not None:
                action = policy(observation, agent)
            else:
                action = 0
            env.step(action)
            env.render()
            if done:
                break

def policy(observation, agent):
    action = random.choice(np.flatnonzero(observation['action_mask']))
    return action


class PettingzooClassic_Iterate2Parallel():
    """
    PettingZoo 1.8.1 version does not provide Parallel API for classic game, 
    so this wrapper is provided for that. 
    But the problem is fixed later: https://github.com/PettingZoo-Team/PettingZoo/issues/379
    """
    def __init__(self, env):
        super(PettingzooClassic_Iterate2Parallel, self).__init__()
        self.env = env
        self.action_spaces = self.env.action_spaces
        self.action_space = list(self.action_spaces.values())[0]
        self.agents = list(self.action_spaces.keys())

        # for rps_v1, discrete to box, fake space
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.uint8)
        # self.observation_spaces = {a:self.observation_space for a in self.agents}

        # for holdem
        obs_space = list(self.env.observation_spaces.values())[0]['observation']
        obs_len = obs_space.shape[0]-self.action_space.n
        self.observation_space = gym.spaces.Box(shape=(obs_len,),low=obs_space.low[:obs_len],high=obs_space.high[:obs_len])
        self.observation_spaces = {a:self.observation_space for a in self.agents}

    def reset(self, observation=None):
        obs = self.env.reset()
        if obs is None:
            return {a:np.zeros(self.observation_space.shape[0]) for a in self.agents}
        else:
            return {a: obs[a]['observation'] for a in self.agents}

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()

    def step(self, action_dict):
        obs_dict, reward_dict, done_dict, info_dict = {}, {}, {}, {}
        for agent, action in action_dict.items():
            observation, reward, done, info = self.env.last()
            valid_actions = np.where(observation['action_mask'])[0]
            if done: 
                action = None  # for classic game: if one player done (requires to set action None), another is not, it causes problem when using parallel API
            elif action not in valid_actions:
                action = random.choice(valid_actions)
            self.env.step(action)
            obs_dict[agent] = observation
             # the returned done from env.last() does not work; reward is for the last step
            # reward_dict[agent] = reward
            # done_dict[agent] = done
            reward_dict[agent] = self.env.rewards[agent]
            done_dict[agent] = self.env.dones[agent]
            info_dict[agent] = info
        return obs_dict, reward_dict, done_dict, info_dict

def run_parallel(args):
    """
    env.step(a1, a2, ...) function takes the actions of all agents all at once.
    # """
    if args.env in ['leduc_holdem_v3', 'texas_holdem_v3']:
        env = eval(args.env).env() 
        env = PettingzooClassic_Iterate2Parallel(env)
    else:
        env = eval(args.env).parallel_env()

    for _ in range(3):
        observation = env.reset()
        # for t in range(1000):
        while True:
            actions = {agent: 1 for agent in env.agents}
            # actions = {'player_0': 0, 'player_1': 1}
            # actions = {n: random.choice(np.arange(env.action_space.n)) for n in ['player_0', 'player_1']}
            observation, reward, done, info = env.step(actions)
            # print(actions, reward, done, info)
            env.render()  # not sure how to render in parallel
            # action = policy(observation, agent)
            if any(done.values()):
                break

def run_parallel2(args):
    """
    Test parallel mode with supersuit env wrappers. 
    """
    parallel_env = eval(args.env).parallel_env()
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(parallel_env, 2)

    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

    # skip frames for faster processing and less control
    # to be compatable with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)

    # downscale observation for faster processing
    env = supersuit.resize_v0(env, 84, 84)

    # allow agent to see everything on the screen despite Atari's flickering screen problem
    parallel_env = supersuit.frame_stack_v1(env, 4)
    parallel_env.seed(1)

    observations = parallel_env.reset()
    print(parallel_env.agents)
    max_cycles = 500
    for step in range(max_cycles):
        actions = {agent: 1 for agent in parallel_env.agents}
        observations, rewards, dones, infos = parallel_env.step(actions)
        parallel_env.render()


def get_args():
    parser = argparse.ArgumentParser(description='Train or test arguments.')
#     parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--env', type=str, help='Environment', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    run_parallel(args)

    # if args.env in ['rps_v1', 'leduc_holdem_v3', 'texas_holdem_v3', 'tictactoe_v3']:
    #     run_iterate(args, policy)
    # else:
    #     run_iterate(args)

