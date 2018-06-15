#!/usr/bin/env python

"""
This script records video

python run_video.py --gamma=0.995 --lam=0.98 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=2 --seed=0 --hid_sizes=16,16 --timesteps_per_batch=2300 --env=ArmDOF_0-v0 --outfile=./result
"""

from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym
from gym.wrappers import Monitor
import numpy as np

import armDOF_0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)    
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    
    env = make(args.env)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    env = Monitor(env, directory=mondir, video_callable=None)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    cfg = args.__dict__
    np.random.seed(args.seed)
    agent = agent_ctor(env.observation_space, env.action_space, cfg)

    # Read File
    weights = np.loadtxt("ArmTrainingResult.txt")
    agent.set_from_flat(weights)
    # Rollout
    ob = env.reset()
    for _ in range(150):
        env.render()
        ob = agent.obfilt(ob)
        action, _ = agent.act(ob)
        env.step(action)

    env.close()
