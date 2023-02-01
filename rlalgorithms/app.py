# -*- coding: utf-8 -*-
"""
Main console script for the silkmoth simulator
"""

import argparse
import sys
import os
import glob
import logging
import datetime
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque

# from bombyxsim.utils import colorize
# from bombyxsim.utils.geometry import Point
from src.utils import fileIO
from src import simulator
from src.agents import silkmoth
from src.controllers import silkmoth_irl, programmed_behavior
from src.envs import smoke_video
# from rlalgorithms.tempdiff import qlearning
import gym
from policygradient import actorcritic

def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="A simulator for olfactory searches")

    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        dest="input_dir",
        help='Path of the directory with odor plume data',
        required=True)
    parser.add_argument(
        "--algorithm",
        type=str,
        dest="algo",
        help='RL algorithm: [QL for Q-learning; Q2L for Double Q-learning]',
        required=True)
    parser.add_argument(
        "-a", "--agent",
        type=str,
        dest="agt",
        help='Type of agent: [silkmoth, ]',
        required=True)
    parser.add_argument(
        "-e", "--env",
        type=str,
        dest="env",
        help='Type of environment: [smokevid, ]',
        required=True)
    parser.add_argument(
        "-c", "--controller",
        type=str,
        dest="controller",
        help='Type of controller: [KPB (Kanzaki et al. programmed behavior), IRL (specify policy file)]',
        nargs='+',
        required=True)
    parser.add_argument(
        "--runs",
        type=int,
        dest="Nruns",
        help='Number of simulation runs',
        required=True)
    parser.add_argument(
        "-T", "--tlim",
        type=int,
        dest="tlim",
        help='Simulation time limit in seconds',
        default=120)
    parser.add_argument(
        "-A", "--animation",
        dest="animation",
        help='Draw animation',
        action="store_true",
        default=0)
    parser.add_argument(
        "--plot-traj",
        dest="plt_traj",
        help='Plot trajectories',
        action="store_true",
        default=0)
    parser.add_argument(
        "--save-log",
        dest="save_log",
        help='Save log to csv',
        action="store_true",
        default=0)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)
    print('Starting script')

    # config_file = os.path.join(args.input_dir, args.env, 'config.json')
    config_file = "/home/nhat/ICT/CPT-IRL/rlalgorithms/examples/smokevid/config.json"
    plumes = [
        i for i in glob.glob(os.path.join(args.input_dir, args.env, '*.h5'))
    ]
    tlim = args.tlim
    env = args.env
    agt = args.agt
    controller = args.controller
    algo = args.algo
    print('RL algorithm: {}'.format(algo))

    g = gym.Gym(args.input_dir,
                config_file,
                args.loglevel,
                tlim,
                env=env,
                agt=agt,
                controller=controller)

    if args.save_log or args.plt_traj:
        out_dir = '{}_{}_{}_{}runs_{}'.format(agt, env, controller[0], args.Nruns, fileIO.tstamp())
        tmp_dir = fileIO.make_dir(args.input_dir, out_dir)

    if args.algo == 'QL':
        ql_policy, stats, p = g.Qlearning(args.Nruns, plumes)

        ql_policy.to_csv('QL-policy-revised.csv')
        # print(ql_policy)
        fig, ax = plt.subplots()
        print('Mean reward per episode:{}'.format(np.mean(
            stats.rewards)))
        ax.plot(pd.Series(stats.rewards).rolling(20).mean())
        plt.savefig('QL-episode-rewards', dpi=300)
        print('Success rate: {:.4f} +- {:.4f}'.format(
            p['success_rate'].sum() / args.Nruns, p['success_rate'].std()))
        print('Search time: {:.4f} +- {:.4f}'.format(
            p.loc[p.success_rate == 1, 'search_time'].mean(),
            p.loc[p.success_rate == 1, 'search_time'].std()))

    elif args.algo == 'SARSA':
        Q, stats, p = g.SARSA(args.Nruns, plumes)
        fig, ax = plt.subplots()
        print('Mean reward per episode:{}'.format(np.mean(stats.rewards)))
        ax.plot(pd.Series(stats.rewards).rolling(20).mean())
        plt.savefig('SARSA-episode-rewards', dpi=300)
        print('Success rate: {:.4f} +- {:.4f}'.format(
        p['success_rate'].sum() / args.Nruns, p['success_rate'].std()))
        print('Search time: {:.4f} +- {:.4f}'.format(
            p.loc[p.success_rate == 1, 'search_time'].mean(),
            p.loc[p.success_rate == 1, 'search_time'].std()))

    elif args.algo == 'AC':
        stats, p, policy = actorcritic.run(g, plumes, args.Nruns)
        print("Performance")
        print('Success rate: {:.4f} +- {:.4f}'.format(
            p['success_rate'].sum() / args.Nruns, p['success_rate'].std()))
        print('Search time: {:.4f} +- {:.4f}'.format(
            p.loc[p.success_rate == 1, 'search_time'].mean(),
            p.loc[p.success_rate == 1, 'search_time'].std()))
        policy.to_csv('actorcritic-policy.csv', index=False)
        # print(policy)
        fig, ax = plt.subplots()
        ax.plot(pd.Series(stats.rewards).rolling(5).mean())
        plt.savefig('episode-rewards', dpi=300)

    # dfQ = pd.DataFrame(Q).T
    # dfQ.sort_index(inplace=True)
    # print(dfQ.describe())
    # pd.DataFrame(full_policy).to_csv('actorcritic-fullpolicy.csv', index=False)
    # print('Q {}:\n{}'.format(Q.shape, Q))
    # print('Performance metrics:')
    # p = pd.DataFrame(g.performance)
    # print('End of script')  

if __name__ == "__main__":
    main(sys.argv[1:])