# from pettingzoo.butterfly import knights_archers_zombies_v7
# env = knights_archers_zombies_v7.env()
from pettingzoo.atari import boxing_v1
from utils.wrappers import PettingZooWrapper, make_env
import supersuit
from utils.env import DummyVectorEnv, SubprocVectorEnv
import gym
import numpy as np

def run_iterate():
    env = boxing_v1.env()
    for _ in range(3):
        env.reset()
        step = 0
        for agent in env.agent_iter():
            step+=1
            print(step)
            observation, reward, done, info = env.last()
            action = 1
            # action = policy(observation, agent)
            env.step(action)
            env.render()
            if done:
                break

def run_parallel():
    # env = boxing_v1.env()
    env = boxing_v1.parallel_env()
    env = PettingZooWrapper(env)
    for _ in range(3):
        env.reset()
        step = 0
        for t in range(1000):
            print(step)
            action = 1
            observation, reward, done, info = env.step(action, action)
            env.render()  # not sure how to render in parallel
 
            # action = policy(observation, agent)
            if done:
                break

def run_parallel2():
    parallel_env = boxing_v1.parallel_env()
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


def test_gym(task, VectorEnv):
    """ 
    Test env parallel DummyVectorEnv (no multiprocess) & SubprocVectorEnv (multiprocess) for single agent gym games.
    """
    env_num = 2
    envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
    assert len(envs) == env_num
    # envs.seed(2)  # which is equal to the next line
    envs.seed(np.random.randint(1000, size=env_num).tolist())
    # envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
    obs = envs.reset()  # reset all environments
    # obs = envs.reset([0, 5, 7])  # reset 3 specific environments
    for i in range(100):
        obs, rew, done, info = envs.step([1] * env_num)  # step synchronously
        envs.render()  # render all environments
    envs.close()  # close all environments

def test_marl(task, VectorEnv, obs_type='ram'):
    """ 
    Test env parallel DummyVectorEnv (no multiprocess) & SubprocVectorEnv (multiprocess) for multi-agent pettingzoo games.
    """
    # env = eval(task).parallel_env(obs_type=obs_type)
    env_num = 2
    envs = VectorEnv([lambda: make_env(task, obs_type=obs_type) for _ in range(env_num)])

    assert len(envs) == env_num
    # envs.seed(2)  # which is equal to the next line
    envs.seed(np.random.randint(1000, size=env_num).tolist())
    # envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
    obs = envs.reset()  # reset all environments
    # obs = envs.reset([0, 5, 7])  # reset 3 specific environments
    for i in range(3000):
        actions = [{'first_0':1, 'second_0':1} for i in range(env_num)]
        obs, r, done, info = envs.step(actions)  # step synchronously
        envs.render()  # render all environments
    envs.close()  # close all environments

if __name__ == '__main__':
    # run_parallel2()
    # run_iterate()

    VectorEnv = [DummyVectorEnv, SubprocVectorEnv][1]
    test_gym('CartPole-v0', VectorEnv)
    # test_marl('slimevolley_v0', VectorEnv)
    # test_marl('pong_v1', VectorEnv, 'ram')


