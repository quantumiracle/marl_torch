# action transformation of SlimeVolley 
action_table = [[0, 0, 0], # NOOP
                [1, 0, 0], # LEFT (forward)
                [1, 0, 1], # UPLEFT (forward jump)
                [0, 0, 1], # UP (jump)
                [0, 1, 1], # UPRIGHT (backward jump)
                [0, 1, 0]] # RIGHT (backward)

max_eps = 50000
max_timesteps = 10000
selfplay_interval = 3000 # interval in a unit of episode to checkpoint a policy and replace its opponent in selfplay

hyperparams = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'lmbda': 0.95,
    'eps_clip': 0.2,
    'hidden_dim': 64,
    'K_epoch': 4,
}
