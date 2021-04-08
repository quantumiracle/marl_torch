import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import numpy as np
from utils.networks import PolicyMLP, ValueMLP, PolicyCNN, ValueCNN

class PPODiscrete(nn.Module):
    def __init__(self, state_space, action_space, func_approx = 'MLP', learner_args={}, **kwargs):
        super(PPODiscrete, self).__init__()
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.eps_clip = kwargs['eps_clip']
        self.K_epoch = kwargs['K_epoch']
        self.device = torch.device(learner_args['device'])
        hidden_dim = kwargs['hidden_dim']

        self.data = []
        if func_approx == 'MLP':
            self.policy = PolicyMLP(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old = PolicyMLP(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())

            self.value = ValueMLP(state_space, hidden_dim).to(self.device)

        elif func_approx == 'CNN':
            self.policy = PolicyCNN(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old = PolicyCNN(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())

            self.value = ValueCNN(state_space, hidden_dim).to(self.device)
        else:
            raise NotImplementedError

        # cannot use lambda in multiprocessing
        # self.pi = lambda x: self.policy.forward(x, softmax_dim=-1)
        # self.v = lambda x: self.value.forward(x)            

        # TODO a single optimizer for two nets may be problematic
        self.optimizer = optim.Adam(list(self.value.parameters())+list(self.policy.parameters()), lr=self.learning_rate, betas=(0.9, 0.999))
        self.mseLoss = nn.MSELoss()

    def pi(self, x):
        return self.policy.forward(x, softmax_dim=-1)

    def v(self, x):
        return self.value.forward(x)  

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                          torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self, GAE=False):
        s, a, r, s_prime, done_mask, oldlogprob = self.make_batch()

        if not GAE:
            rewards = []
            discounted_r = 0
            for reward, is_continue in zip(reversed(r), reversed(done_mask)):
                if not is_continue:
                    discounted_r = 0
                discounted_r = reward + self.gamma * discounted_r
                rewards.insert(0, discounted_r)  # insert in front, cannot use append

            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for _ in range(self.K_epoch):
            vs = self.v(s)

            if GAE:
                # use generalized advantage estimation
                vs_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = vs_target - self.v(s)
                delta = delta.detach()

                advantage_lst = []
                advantage = 0.0
                for delta_t in torch.flip(delta, [0]):
                    advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                    advantage_lst.append(advantage)
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            else:
                advantage = rewards - vs.squeeze(dim=-1).detach()
                vs_target = rewards

            pi = self.pi(s)
            dist = Categorical(pi)
            dist_entropy = dist.entropy()
            logprob = dist.log_prob(a)
            # pi_a = pi.gather(1,a)
            
            # ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            ratio = torch.exp(logprob - oldlogprob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(vs , vs_target.detach()) - 0.01*dist_entropy
            loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def choose_action(self, s, Greedy=False):
        prob = self.policy_old(torch.from_numpy(s).unsqueeze(0).float().to(self.device)).squeeze()  # make sure input state shape is correct
        if Greedy:
            a = torch.argmax(prob, dim=-1).item()
            return a
        else:
            dist = Categorical(prob)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a.detach().item(), logprob.detach().item()

    def save_model(self, path=None):
        torch.save(self.policy.state_dict(), path+'_policy')
        torch.save(self.value.state_dict(), path+'_value')


    def load_model(self, path=None):
        self.policy.load_state_dict(torch.load(path+'_policy'))
        self.policy_old.load_state_dict(self.policy.state_dict())  # important

        self.value.load_state_dict(torch.load(path+'_value'))


class MultiPPODiscrete(nn.Module):
    """ 
    Multi-agent PPO for a single discrete environment.
    Input: observation (dict) {agent_name: agent_observation}
    Output: action (dict) {agent_name: agent_action}
    """
    def __init__(self, agents, state_spaces, action_spaces, func_approx = 'MLP', fixed_agents=[], learner_args={}, **kwargs):
        super(MultiPPODiscrete, self).__init__()
        self.fixed_agents = fixed_agents
        self.agents = {}
        for agent_name, state_space, action_space in zip(agents, state_spaces.values(), action_spaces.values()):
            self.agents[agent_name] = PPODiscrete(state_space, action_space, func_approx, learner_args, **kwargs).to(learner_args['device'])
        # validation check 
        for agent in fixed_agents:
            assert agent in self.agents

    def put_data(self, transition):
        (observations, actions, rewards, observations_, logprobs, dones) = transition
        data = (observations.values(), actions.values(), rewards.values(), observations_.values(), logprobs.values(), dones.values())
        for agent_name, *sample in zip(self.agents, *data):
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].put_data(tuple(sample))
        
    def make_batch(self):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].make_batch()

    def train_net(self, GAE=False):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                # print('trained agents: ', agent_name)
                self.agents[agent_name].train_net(GAE)

    def choose_action(self, observations, Greedy=False):
        actions={}
        logprobs={}
        for agent_name in self.agents:
            actions[agent_name], logprobs[agent_name] = self.agents[agent_name].choose_action(observations[agent_name], Greedy)
        return actions, logprobs

    def save_model(self, path=None):
        for idx, agent_name in enumerate(self.agents):
            if agent_name not in self.fixed_agents:

                self.agents[agent_name].save_model(path+'_{}'.format(idx))

    def load_model(self, agent_name=None, path=None):
        if agent_name is not None:  # load model for specific agent only
            self.agents[agent_name].load_model(path)
        else:
            for idx, agent_name in enumerate(self.agents):
                self.agents[agent_name].load_model(path+'_{}'.format(idx))


class ParallelPPODiscrete(nn.Module):
    """
    PPO handles parallel envs wrapped with ***VectorEnv wrapper.
    """
    def __init__(self, num_envs, state_space, action_space, func_approx = 'MLP', learner_args={}, **kwargs):
        super(ParallelPPODiscrete, self).__init__()
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.eps_clip = kwargs['eps_clip']
        self.K_epoch = kwargs['K_epoch']
        self.device = torch.device(learner_args['device'])
        hidden_dim = kwargs['hidden_dim']
        self.num_envs = num_envs

        self.data = [[] for _ in range(self.num_envs)]
        if func_approx == 'MLP':
            self.policy = PolicyMLP(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old = PolicyMLP(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())

            self.value = ValueMLP(state_space, hidden_dim).to(self.device)

        elif func_approx == 'CNN':
            self.policy = PolicyCNN(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old = PolicyCNN(state_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())

            self.value = ValueCNN(state_space, hidden_dim).to(self.device)
        else:
            raise NotImplementedError

        # cannot use lambda in multiprocessing
        # self.pi = lambda x: self.policy.forward(x, softmax_dim=-1)
        # self.v = lambda x: self.value.forward(x)            

        # TODO a single optimizer for two nets may be problematic
        self.optimizer = optim.Adam(list(self.value.parameters())+list(self.policy.parameters()), lr=self.learning_rate, betas=(0.9, 0.999))
        self.mseLoss = nn.MSELoss()

    def pi(self, x):
        return self.policy.forward(x, softmax_dim=-1)

    def v(self, x):
        return self.value.forward(x)  

    def put_data(self, transition):
        for i, trans in enumerate(transition): # iterate over the list of envs
            self.data[i].append(trans)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        batch_data = []
        for env_data in self.data:  # iterate over the list of envs
            for transition in env_data:
                s, a, r, s_prime, prob_a, done = transition
                
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                prob_a_lst.append(prob_a)
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                
            s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                            torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                            torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)
        batch_data.append([s,a,r,s_prime,done_mask, prob_a])    
        self.data = [[] for _ in range(self.num_envs)]
        return batch_data
        
    def train_net(self, GAE=False):
        for data in self.make_batch():  # iterate over the list of envs
            s, a, r, s_prime, done_mask, oldlogprob = data

            if not GAE:
                rewards = []
                discounted_r = 0
                for reward, is_continue in zip(reversed(r), reversed(done_mask)):
                    if not is_continue:
                        discounted_r = 0
                    discounted_r = reward + self.gamma * discounted_r
                    rewards.insert(0, discounted_r)  # insert in front, cannot use append

                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            for _ in range(self.K_epoch):
                vs = self.v(s)

                if GAE:
                    # use generalized advantage estimation
                    vs_target = r + self.gamma * self.v(s_prime) * done_mask
                    delta = vs_target - self.v(s)
                    delta = delta.detach()

                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta, [0]):
                        advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

                else:
                    advantage = rewards - vs.squeeze(dim=-1).detach()
                    vs_target = rewards

                pi = self.pi(s)
                dist = Categorical(pi)
                dist_entropy = dist.entropy()
                logprob = dist.log_prob(a)
                # pi_a = pi.gather(1,a)
                
                # ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
                ratio = torch.exp(logprob - oldlogprob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(vs , vs_target.detach()) - 0.01*dist_entropy
                loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def choose_action(self, s, Greedy=False):
        prob = self.policy_old(torch.from_numpy(s).float().to(self.device)).squeeze()  # make sure input state shape is correct
        if Greedy:
            a = torch.argmax(prob, dim=-1).item()
            return a
        else:
            dist = Categorical(prob)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a.detach().cpu().numpy(), logprob.detach().cpu().numpy()

    def save_model(self, path=None):
        torch.save(self.policy.state_dict(), path+'_policy')
        torch.save(self.value.state_dict(), path+'_value')


    def load_model(self, path=None):
        self.policy.load_state_dict(torch.load(path+'_policy'))
        self.policy_old.load_state_dict(self.policy.state_dict())  # important

        self.value.load_state_dict(torch.load(path+'_value'))

class ParallelMultiPPODiscrete(nn.Module):
    """
    Multi-agent PPO for multiple parallel discrete environments (wrapped with ***VectorEnv wrapper).
    Input: observation (list of dict) [{agent_name: agent_observation}, {agent_name: agent_observation}], each dict for a single env;
    Output: action (list of dict) [{agent_name: agent_action}, {agent_name: agent_observation}], each dict for a single env.
    """
    def __init__(self, num_envs, agents, state_spaces, action_spaces, func_approx = 'MLP', fixed_agents=[], learner_args={}, **kwargs):
        super(ParallelMultiPPODiscrete, self).__init__()
        self.fixed_agents = fixed_agents
        self.num_envs = num_envs
        self.agents = {}
        for agent_name, state_space, action_space in zip(agents, state_spaces.values(), action_spaces.values()):
            self.agents[agent_name] = ParallelPPODiscrete(num_envs, state_space, action_space, func_approx, learner_args, **kwargs).to(learner_args['device'])
        # validation check 
        for agent in fixed_agents:
            assert agent in self.agents

    def put_data(self, transition):
        (observations, actions, rewards, observations_, logprobs, dones) = transition
        
        data = [[] for _ in range(len(self.agents))]  # first dimension is the agent
        for obs, act, r, obs_, logp, d in zip(observations, actions, rewards, observations_, logprobs, dones): # iterate over envs
            env_data = (obs.values(), act.values(), r.values(), obs_.values(), logp.values(), d.values())
            for j, *agent_data in enumerate(zip(*env_data)): # iterate over agents
                data[j].append(*agent_data)  # *agent_data is a tuple

        for agent_name, *sample in zip(self.agents, *data):
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].put_data(tuple(sample))
        
    def make_batch(self):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].make_batch()

    def train_net(self, GAE=False):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                # print('trained agents: ', agent_name)
                self.agents[agent_name].train_net(GAE)

    def choose_action(self, observations, Greedy=False):
        multi_actions = []
        multi_logprobs = []

        # looping through both the envs and agents is slow
        # for obs in observations: # iterate over the list of envs
        #     actions={}
        #     logprobs={}
        #     for agent_name in self.agents:
        #         actions[agent_name], logprobs[agent_name] = self.agents[agent_name].choose_action(obs[agent_name], Greedy)
        #     multi_actions.append(actions)
        #     multi_logprobs.append(logprobs)
        # return multi_actions, multi_logprobs

        obs = np.array([list(obs.values()) for obs in observations]).swapaxes(0,1) # shape after swap: (# agents, # envs, obs_dim)
        for i, agent_name in enumerate(self.agents):
                actions, logprobs = self.agents[agent_name].choose_action(obs[i], Greedy)  # one forward inference for multiple envs
                multi_actions.append(actions)  # multi_actions shape: (# agents, # envs)
                multi_logprobs.append(logprobs)

        actions_list, logprobs_list = [{} for _ in range(self.num_envs)], [{} for _ in range(self.num_envs)]
        for i, agent_name in enumerate(self.agents):
            for j in range(self.num_envs):
                actions_list[j][agent_name] = multi_actions[i][j]
                logprobs_list[j][agent_name] = multi_logprobs[i][j]

        return actions_list, logprobs_list

    def save_model(self, path=None):
        for idx, agent_name in enumerate(self.agents):
            if agent_name not in self.fixed_agents:
                
                self.agents[agent_name].save_model(path+'_{}'.format(idx))

    def load_model(self, agent_name=None, path=None):
        if agent_name is not None:  # load model for specific agent only
            self.agents[agent_name].load_model(path)
        else:
            for idx, agent_name in enumerate(self.agents):
                self.agents[agent_name].load_model(path+'_{}'.format(idx))



