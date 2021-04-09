import argparse
import torch 

def get_args():
    parser = argparse.ArgumentParser(description='Train or test arguments.')
#     parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--env', type=str, 
                        help='Environment', required=True)
    parser.add_argument('--ram', dest='ram_obs', action='store_true', default=False,
                        help='RAM observation rather than RGB images')
    parser.add_argument('--render', action='store_true',
                        help='Enable openai gym real-time rendering')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of environments for parallel sampling')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--alg', type=str, default='td3',
                        help='Choose algorithm type')
    parser.add_argument('--selfplay', action='store_true', default=False, 
                        help='The selfplay mode')
    parser.add_argument('--load-agent', type=str, default=None, 
                        help='Load agent models by specifying: 1, 2, or both')
    parser.add_argument('--train-both', action='store_true', default=False, 
                        help='Train both agents rather than train the second player only as default')
    parser.add_argument('--against-baseline', action='store_true', default=False,
                        help='Let the agent play against the baseline (given by the environment)')
    parser.add_argument('--fictitious', action='store_true', default=False,
                        help='Use fictitious self play to train the agent')    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load pre-trained model')  
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


    return args