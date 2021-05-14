"""
Testing script for PettingZoo environments.
PettingZoo as a multi-agent library provides two environment interaction modes:
1. iterate: the action of each agent is taken iteratively in order with env.step(ai) function;
2. parallel: the actions of all agents are taken at the same env.step(a1, a2, ...) function.
"""

# from pettingzoo.butterfly import knights_archers_zombies_v7
# env = knights_archers_zombies_v7.env()
from pettingzoo.atari import boxing_v1
from utils.wrappers import PettingZooWrapper
import supersuit

def run_iterate():
    """
    env.step(a) function iteratively takes the action of each agent.
    """
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
    """
    env.step(a1, a2, ...) function takes the actions of all agents all at once.
    """
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
    """
    Test parallel mode with supersuit env wrappers. 
    """
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


if __name__ == '__main__':
    run_parallel2()
    # run_iterate()

