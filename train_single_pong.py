import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import numpy as np
from utils.ppo import PPODiscrete
from gym import spaces
from collections import deque

class ScaledFloatFrame(gym.ObservationWrapper):
     def observation(self, obs):
         return np.array(obs).astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser(description='Train or test arguments.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--render', dest='render', action='store_true',
                    help='Enable openai gym real-time rendering')
    args = parser.parse_args()
    
    SEED = 721
    env = gym.make('Pong-ram-v4')   # CartPole-v1 
    env = ScaledFloatFrame(env)  # scaled observation, this is essential!
    # env = gym.make('LunarLander-v2')   
    max_length = 10000
    max_episode = 500000
    stack = 1  # stack frames/ram
    env.seed(SEED)
    # state_space = env.observation_space
    state_space = spaces.Box(0, 255, (stack*128,))
    action_space = env.action_space
    print(state_space, action_space)

    hyperparams = {
        'learning_rate': 3e-3,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'hidden_dim': 64,
        'K_epoch': 4,
    }
    device_idx = 0
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    learner_args = {'device':  device}

    model = PPODiscrete(state_space, action_space, 'MLP', learner_args, **hyperparams).to(device)
    if args.test:
        model.load_model('model/mappo')

    score = 0.0
    print_interval = 20
    save_interval = 100
    epi_len = []
    s_queue = deque([], maxlen=stack)
    for n_epi in range(max_episode):
        s = env.reset()
        for _ in range(stack):
            s_queue.append(s)
        done = False
        for t in range(max_length):
            stack_s = np.array(s_queue).reshape(-1)
            a, logprob = model.choose_action(stack_s)
            s_prime, r, done, info = env.step(a)  # from discrete to multibinary action
            s_queue.append(s_prime)
            stack_s_prime = np.array(s_queue).reshape(-1)
            if args.render:
                env.render()
            model.put_data((stack_s, a, r, stack_s_prime, logprob, done))
            # s = s_prime

            score += r
            if done:
                break
        if args.train:
            model.train_net()
        epi_len.append(t)
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.3f}, avg epi length : {}".format(n_epi, score/print_interval, int(np.mean(epi_len))))
            score = 0.0
            epi_len = []
        if n_epi%save_interval==0 and n_epi!=0 and args.train:
            model.save_model('model/mappo')
    env.close()

if __name__ == '__main__':
    main()
