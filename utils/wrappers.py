import pettingzoo
import slimevolleygym  # https://github.com/hardmaru/slimevolleygym
import supersuit  # wrapper for pettingzoo envs
import gym
from gym import spaces


AtariEnvs = ['basketball_pong_v1', 'boxing_v1', 'combat_plane_v1', 'combat_tank_v1',
 'double_dunk_v2', 'entombed_competitive_v2', 'entombed_cooperative_v2', 'flag_capture_v1', 
 'foozpong_v1', 'ice_hockey_v1', 'joust_v2', 'mario_bros_v2', 'maze_craze_v2', 'othello_v2',
  'pong_v1', 'quadrapong_v2', 'space_invaders_v1', 'space_war_v1', 'surround_v1', 'tennis_v2', 
  'video_checkers_v3', 'volleyball_pong_v1', 'warlords_v2', 'wizard_of_wor_v2']

AtariTwoPlayerCompetitiveEnvs = ['basketball_pong_v1', 'boxing_v1', 'combat_plane_v1', 'combat_tank_v1',
 'double_dunk_v2', 'entombed_competitive_v2', 'flag_capture_v1', 'joust_v2', 'maze_craze_v2', 'othello_v2',
 'pong_v1', 'space_war_v1', 'surround_v1', 'tennis_v2', 'video_checkers_v3']

AtariTwoPlayerCooperativeEnvs = ['entombed_cooperative_v2']

AtariTwoPlayerMixedSumEnvs = ['joust_v2', 'mario_bros_v2', 'space_invaders_v1']

AtariMoreThan2PlayersEnvs = ['foozpong_v1',  # 4 players, 2 teams
'quadrapong_v2', # 4 players, 2 teams
'volleyball_pong_v1', # 4 players, 2 teams
'warlords_v2', # 4 players, competitive
'wizard_of_wor_v2' # 4 players, competitive and coorperative
]

# import envs: multi-agent environments in PettingZoo Atari (both competitive and coorperative)
for env in AtariEnvs:   
    exec("from pettingzoo.atari import {}".format(env))
        
# class PettingZooWrapper(pettingzoo.utils.wrappers.OrderEnforcingWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         self.a1 = 'first_0'
#         self.a2 = 'second_0'
#         self.observation_space = self.env.observation_spaces[self.a1]
#         self.action_space = self.env.action_spaces[self.a1]
#         self.reward_range = (-1000000, 1000000)
#         fake_env = gym.make('Pong-v0')
#         self.spec = fake_env.spec
#         fake_env.close()
#         # self.unwrapped = self.env.aec_env.env.env.env.env
#     def step(self, action1, action2):
#         actions = {self.a1: action1, self.a2: action2}
#         next_state, reward, done, info = self.env.step(actions)
#         # modify ...
#         return next_state[self.a1], reward[self.a1], done[self.a1], info[self.a1]

#     def reset(self, observation=None):
#         return self.env.reset()

class PettingZooWrapper(pettingzoo.utils.wrappers.OrderEnforcingWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, observation=None):
        return self.env.reset()

class SlimeVolleyWrapper():
    # action transformation of SlimeVolley 
    action_table = [[0, 0, 0], # NOOP
                    [1, 0, 0], # LEFT (forward)
                    [1, 0, 1], # UPLEFT (forward jump)
                    [0, 0, 1], # UP (jump)
                    [0, 1, 1], # UPRIGHT (backward jump)
                    [0, 1, 0]] # RIGHT (backward)


    def __init__(self, env):
        super(SlimeVolleyWrapper, self).__init__()
        self.env = env
        self.agents = ['first_0', 'second_0']
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = spaces.Discrete(len(self.action_table))
        self.action_spaces = {name: self.action_space for name in self.agents}


    def reset(self, observation=None):
        obs1 = self.env.reset()
        obs2 = obs1 # both sides always see the same initial observation.

        obs = {}
        obs[self.agents[0]] = obs1
        obs[self.agents[1]] = obs2
        return obs

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def step(self, actions, against_baseline=False):
        obs, rewards, dones, infos = {},{},{},{}
        actions_ = [self.env.discreteToBox(a) for a in actions.values()]  # from discrete to multibinary action

        if against_baseline:
            # this is for validation: load a single policy as 'second_0' to play against the baseline agent (via self-play in 2015)
            obs2, reward, done, info = self.env.step(actions_[1]) # extra argument
            obs1 = obs2 
            rewards[self.agents[0]] = -reward
            rewards[self.agents[1]] = reward # the reward is for the learnable agent (second)
        else:
            # normal 2-player setting
            obs1, reward, done, info = self.env.step(*actions_) # extra argument
            obs2 = info['otherObs']
            rewards[self.agents[0]] = reward
            rewards[self.agents[1]] = -reward

        obs[self.agents[0]] = obs1
        obs[self.agents[1]] = obs2
        dones[self.agents[0]] = done
        dones[self.agents[1]] = done
        infos[self.agents[0]] = info
        infos[self.agents[1]] = info

        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

def make_env(env_name='boxing_v1', seed=1, obs_type='rgb_image'):
    '''https://www.pettingzoo.ml/atari'''
    if env_name == 'slimevolley_v0':
        env = SlimeVolleyWrapper(gym.make("SlimeVolley-v0"))

    else: # PettingZoo envs
        env = eval(env_name).parallel_env(obs_type=obs_type)

        if obs_type == 'rgb_image':
            # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
            # to deal with frame flickering
            env = supersuit.max_observation_v0(env, 2)

            # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
            env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

            # skip frames for faster processing and less control
            # to be compatable with gym, use frame_skip(env, (2,5))
            env = supersuit.frame_skip_v0(env, 4)

            # downscale observation for faster processing
            env = supersuit.resize_v0(env, 84, 84)

            # allow agent to see everything on the screen despite Atari's flickering screen problem
            env = supersuit.frame_stack_v1(env, 4)

        else:
            env = supersuit.frame_skip_v0(env, 4)

        #   env = PettingZooWrapper(env)  # need to be put at the end
        if env_name in AtariEnvs:  # normalize the observation of Atari for both image or RAM 
            env = supersuit.dtype_v0(env, 'float32') # need to transform uint8 to float first for normalizing observation: https://github.com/PettingZoo-Team/SuperSuit
            env = supersuit.normalize_obs_v0(env, env_min=0, env_max=1) # normalize the observation to (0,1)

        # assign observation and action spaces
        env.observation_space = list(env.observation_spaces.values())[0]
        env.action_space = list(env.action_spaces.values())[0]

    env.seed(seed)
    return env
