# Command Instruction

### Single Agent

For training PPO against the environment baseline:

1. `python train_single_pong.py --train`


2. `python train_single_slimevolley.py --train`

### Two Agents

1. For PettingZoo or SlimeVolley:

   `python train_pettingzoo_mp_vecenv.py --env pong_v1 --ram --num-envs 3 --selfplay`

