import os
import datetime
import time

def create_log_dir(args):
    log_dir = ""
    log_dir = log_dir + "{}-".format(args.env)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now
    log_dir = os.path.join("runs", log_dir)
    return log_dir

def load_model(model, args):
    if args.load_agent=='1':  # load a pretrained model as opponent
        model.load_model(agent_name='first_0', path=f'model/{args.env}/mappo')
    elif args.load_agent=='2':  # load a pretrained model as opponent
        model.load_model(agent_name='second_0', path=f'model/{args.env}/mappo')
    elif args.load_agent=='both':
        model.load_model(agent_name='first_0', path=f'model/{args.env}/mappo')
        model.load_model(agent_name='second_0', path=f'model/{args.env}/mappo')

    if args.test:
        model.load_model(agent_name='second_0', path=f'model/{args.env}/'+args.load_model)
        model.load_model(agent_name='first_0', path=f'model/{args.env}/'+args.load_model)
