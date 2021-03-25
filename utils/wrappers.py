import pettingzoo
import supersuit  # wrapper for pettingzoo envs
import gym

AtariEnvs = ['basketball_pong_v1', 'boxing_v1', 'combat_plane_v1', 'combat_tank_v1',
 'double_dunk_v2', 'entombed_competitive_v2', 'entombed_cooperative_v2', 'flag_capture_v1', 
 'foozpong_v1', 'ice_hockey_v1', 'joust_v2', 'mario_bros_v2', 'maze_craze_v2', 'othello_v2',
  'pong_v1', 'quadrapong_v2', 'space_invaders_v1', 'space_war_v1', 'surround_v1', 'tennis_v2', 
  'video_checkers_v3', 'volleyball_pong_v1', 'warlords_v2', 'wizard_of_wor_v2']

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


def make_env(env_name='boxing_v1', seed=1, obs_type='rgb_image'):
    '''https://www.pettingzoo.ml/atari'''
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

    #   env = PettingZooWrapper(env)  # need to be put at the end

    env.seed(seed)
    return env
