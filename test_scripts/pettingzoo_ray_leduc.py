from copy import deepcopy
from numpy import float32
import os
from supersuit import dtype_v0

import ray
import argparse
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from pettingzoo.classic import leduc_holdem_v3
import tensorflow as tf

from ray.rllib.models import ModelCatalog

from ray.tune.registry import register_env
from gym.spaces import Box

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchMaskedActions(TorchModelV2, torch.nn.Module):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        torch.nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        obs_len = obs_space.shape[0]-action_space.n

        orig_obs_space = Box(shape=(obs_len,),low=obs_space.low[:obs_len],high=obs_space.high[:obs_len])
        self.action_embed_model = TorchFC(orig_obs_space, action_space, action_space.n,
            model_config, name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]['observation']
        })
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

def train():
    alg_name = "PPO"
    ModelCatalog.register_custom_model(
        "pa_model", TorchMaskedActions)
    # function that outputs the environment you wish to register.
    def env_creator():
        env = leduc_holdem_v3.env()
        return env

    num_cpus = 1

    config = deepcopy(get_agent_class(alg_name)._default_config)

    register_env("leduc_holdem",
                 lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    print(obs_space)
    act_space = test_env.action_space

    config["multiagent"] = {
        "policies": {
            "player_0": (None, obs_space, act_space, {}),
            "player_1": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: agent_id
    }

    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    # config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    config["rollout_fragment_length"] = 30
    config["train_batch_size"] = 200
    config["horizon"] = 200
    config["no_done_at_end"] = False
    config["framework"] = "torch"
    config["model"] = {
        "custom_model":"pa_model",
    }

    # config['hiddens'] = []
    config['env'] = "leduc_holdem"

    ray.init(num_cpus=num_cpus + 1)

    tune.run(
        alg_name,
        name="PPO-leduc_holdem",
        stop={"timesteps_total": 10000000},
        checkpoint_freq=10,
        config=config
    )


train()