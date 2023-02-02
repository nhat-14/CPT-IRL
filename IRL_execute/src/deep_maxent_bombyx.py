#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file estimates the reward function of a silkmoth based on Deep Maximum Entropy IRL and trajectories obtained from wind tunnel experiments.
"""

from myirl.irl.mdp import neo_mothworld
from myirl.irl import value_iteration
from myirl.irl import deep_maxent
from getpass import getpass
from pathlib import Path
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
import h5py
import json
import glob
import os
import argparse
import sys
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="sans-serif", rc={"font.sans-serif": ["FreeSans", "Arial"]})

def tstamp(): 
    return str(datetime.datetime.now().strftime('%m%d_%H%M%S'))


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Estimates the reward function of a silkmoth based on MaxEnt Deep IRL")
    parser.add_argument(
        '--version',
        action='version',
        version='MyIRL {ver}'.format(ver=1))
    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        help='Save extracted reward and action-values as csv',  action="store_true",
        default=0)
    parser.add_argument(
        "--plots",
        dest="save_plots",
        help='Plot reward and action-value function',
        action="store_true",  default=0)
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        dest="input_dir",
        help='Path of the directory with the input data',
        required=True)
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        dest="epochs",
                        help='Training epochs for the gradient descent',
                        default=200)
    parser.add_argument("-t",
                        "--trajectory-length",
                        type=int,
                        dest="traj_len",
                        help='Length in discrete time steps of the expert demonstrations',
                        required=True)
    parser.add_argument("-g",
                        "--discount",
                        type=float,
                        dest="discount",
                        help='Discount factor',
                        default=0.9)
    parser.add_argument("-l",
                        "--learning-rate",
                        type=float,
                        dest="learning_rate",
                        help='Learning rate for gradient optimization',
                        default=0.01)
    parser.add_argument("--l2-reg",
                        dest="l2reg",
                        help='Whether to use L2 regularization in the gradient descent',
                        type=float,
                        nargs="+")
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv', '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    return parser.parse_args(args)


def read_trajectories(csvs, cols, tl=0):

    trajs = []
    # Make a (T, L, 2) 3D numpy array where T is the number of csv files and L is the trajectory length.
    for c in csvs:

        # Read data from csv into pandas dataframe
        df = pd.read_csv(c, index_col=None, nrows=tl)
        df1 = df.iloc[-109:-1]

        print('Working on: {}.csv'.format(
            os.path.basename(os.path.splitext(c)[0])))

        # Handle data as numpy array
        num_df = df[cols].values

        # States
        s = num_df[:, 0]
        s = s.reshape(-1, 1)
        # print('States: \n{}'.format(s))

        # Actions
        a = num_df[:, 1]  # - 1
        a = a.reshape(-1, 1)
        # print('Actions: \n{}'.format(a))

        # Make s and a into columns and concatenate them
        traj = np.concatenate((s.reshape(-1, 1), a), axis=1)
        trajs.append(traj)

    # List of trajectories -> 3D numpy array
    trajs = np.stack(trajs)

    return trajs

def plot_reward_function(df, yticks, save_path, _annot=True):
    """Plot reward function for each state

        Args:
            df (pandas.DataFrame): Data from csv
            yticks (list of str): Labels for y-ticks
        """

    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax = plt.subplots(figsize=(7, 4))

    ax = sns.heatmap(
        df.T,
        # center=0,
        cmap='magma',
        yticklabels=yticks,
        annot=_annot,
        cbar_kws={'label': 'Reward'})
    ax.invert_yaxis()

    ax.set_xlabel('Blank duration')
    ax.set_ylabel('Antennae')

    fig.tight_layout()
    # plt.savefig(save_path, dpi=300)
    plt.show()


def main(args):
    args = parse_args(args)

    # Load and parse json configuration
    config_file = os.path.join(args.input_dir, 'config.json')
    with open(config_file) as f:
        cfg = json.load(f)

    grid = cfg['n_states']
    grid_axes = cfg['n_sub_states']
    discount = args.discount
    epochs = args.epochs
    learning_rate = args.learning_rate
    traj_len = args.traj_len
    state_names = cfg['state_names']
    if args.l2reg:
        l1, l2 = tuple(args.l2reg)
    else:
        l1 = l2 = 0

    # Path where all output data will be stored
    out_dir = 'DeepMaxEnt_e{}_t{}_g{:03d}_{}'.format(epochs, traj_len, round(100 * discount), tstamp())

    # Read csv logs from input_dir into trajectories
    csvs = [i for i in glob.glob(os.path.join(args.input_dir, '[!_]*.csv'))]
    trajectories = read_trajectories(csvs, cfg["df_cols"], tl=traj_len)

    # Load the state transition probabilities
    tp = np.load(os.path.join(args.input_dir, 'trans_prob.npy'))

    # Construct a mothworld object
    mw = neo_mothworld.Mothworld(grid, grid_axes, discount, tp)

    # Initialize the feature matrix
    feature_matrix = mw.load_feature_map(os.path.join(args.input_dir, '_features.csv'))
    print('Shape of feature matrix: {}'.format(feature_matrix.shape))

    # Extract a reward function using MaxEnt IRL and the moth trajectories

    structure = (4, 3)
    print('NN structure: {}; learning rate: {}'.format(structure, learning_rate))
    # Calculating reward
    r = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix, mw.n_actions, mw.discount, mw.transition_probability, trajectories, epochs, learning_rate, l1=l1, l2=l2)

    # Reshape reward
    ex_reward = r.reshape(*grid_axes)
    print('Mean reward: {}'.format([
        round(np.mean(ex_reward[:, i]), 3) for i in range(ex_reward.shape[1])
    ]))

    # Store extracted Q value
    # Calculating policy
    moth_policy = value_iteration.find_policy(mw.n_states, mw.n_actions,
                                              mw.transition_probability, r,
                                              discount, threshold=1e-2)
    
    simple_policy = np.array([np.argmax(moth_policy[i,:]) for i in range(mw.n_states)])
    simple_policy = simple_policy.reshape(mw.substate).T

    # Save policy into csv
    pd.DataFrame(moth_policy).to_csv('raw_policy.csv', index=False)
    moth_policy = moth_policy.T
    print('Policy: {}\n{}'.format(moth_policy.shape, moth_policy))
    action_labels = cfg["action_labels"]

    if args.save_csv:
        Path(os.path.join(args.input_dir, out_dir)).mkdir(
            parents=True, exist_ok=True)
        tmp_dir = os.path.join(args.input_dir, out_dir)

        # Saving reward to csv
        pd.DataFrame(ex_reward).to_csv(os.path.join(tmp_dir, 'Reward.csv'))
        
        np.savetxt(os.path.join(tmp_dir,'simple_policy.csv'), simple_policy, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(tmp_dir,'moth_policy.csv'), moth_policy, delimiter=',')

        # Saving policy to h5 and csv
        with h5py.File(os.path.join(tmp_dir, 'policy.h5'), 'w') as hf:
            hf.create_dataset("policy",  data=moth_policy.reshape(mw.n_actions, *grid_axes))

        for i in range(len(moth_policy)):
            pd.DataFrame(moth_policy[i].reshape(grid_axes)).to_csv(
                os.path.join(tmp_dir, 'policy_{0}.csv'.format(action_labels[i])))

        print('Reward and policy saved to disk')

    if args.save_plots:

        Path(os.path.join(args.input_dir, out_dir)).mkdir(
            parents=True, exist_ok=True)
        tmp_dir = os.path.join(args.input_dir, out_dir)
        plot_title = 'Epochs: {}, Trajectory length: {}, Discount: {}, Learning rate: {}'.format(
            epochs, traj_len, discount, learning_rate)

        sns.set_style('ticks')
        sns.set_context('paper')

        fig, ax_r = plt.subplots(figsize=(7, 4))

        ax_r = sns.heatmap(ex_reward.T,
                           center=0,
                           cmap='magma',
                           yticklabels=cfg["substate_ticks"][state_names[1]],
                           annot=True,
                           cbar_kws={'label': 'Reward'})

        ax_r.set_xlabel(cfg["state_labels"][state_names[0]])
        ax_r.set_ylabel('{}'.format(cfg["state_labels"][state_names[1]]))
        ax_r.invert_yaxis()

        if cfg["draw_subgrid"]:
            ax_r.hlines(cfg["subgrid_ticks"], *ax_r.get_ylim(),
                        linestyle='--', linewidth=0.5, color='w')

        fig.suptitle(plot_title)

        plt.savefig(os.path.join(tmp_dir, 'Reward.png'), dpi=300)
        plt.show()

        fig, ax_q = plt.subplots(
            1, mw.n_actions, figsize=(12, 6.8), sharey=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

        for i, ax in enumerate(ax_q.flat):

            sns.heatmap(moth_policy[i].reshape(*grid_axes), center=0,
                        ax=ax,
                        cbar=i == 0,
                        vmin=0, vmax=1,
                        cbar_ax=None if i else cbar_ax,
                        xticklabels=cfg["substate_ticks"][state_names[1]], annot=True)

            ax.set_title('{}'.format(action_labels[i]))
            ax.invert_yaxis()
            ax.set_xlabel('{}'.format(cfg["state_labels"][state_names[1]]))

            if cfg["draw_subgrid"]:
                ax.vlines(cfg["subgrid_ticks"], *ax.get_ylim(),
                          linestyle='--', linewidth=0.5, color='w')

            ax_q[0].set_ylabel(cfg["state_labels"][state_names[0]])

        fig.suptitle(plot_title)
        plt.savefig(os.path.join(tmp_dir, 'Policy.png'), dpi=300)
        print('Plots saved to disk')

if __name__ == "__main__":
    main(sys.argv[1:])