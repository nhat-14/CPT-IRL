#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file estimates the reward function of a silkmoth based on Maximum Entropy IRL and trajectories obtained from wind tunnel experiments.
"""

import argparse
import sys
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font="sans-serif", rc={"font.sans-serif":["FreeSans","Arial"]})
import os
import glob
import json
from pathlib import Path
from getpass import getpass

from myirl.irl import maxent
from myirl.irl import value_iteration
from myirl.irl.mdp import neo_mothworld


_logger = logging.getLogger(__name__)
tstamp = lambda: str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Estimates the reward function of a silkmoth based on MaxEnt IRL")
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
        "--send-email",
        dest="send_email",
        help='If true, send email notification when script finishes',
        action="store_true",  default=0)
    parser.add_argument(
        "--data-dir", 
        type=str,  
        dest="data_dir",  
        help='Path of the directory with the input data', 
        required=True)
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

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

def read_trajectories(csvs, tl):
    
    trajs = []
    # Make a (T, L, 2) 3D numpy array where T is the number of csv files and L is the trajectory length.
    for c in csvs:

        # Read data from csv into pandas dataframe
        df = pd.read_csv(c, index_col=None, usecols=['Time', 'state_i', 'action'])#, nrows=tl)
        # df = pd.read_csv(c, index_col=None, usecols=['state', 'action'])
      
        _logger.debug('Working on: {}.csv'.format(
            os.path.basename(os.path.splitext(c)[0])))
        
        # Downsample data to trajectory length
        # N = len(df)
        # T0 = round(df.Time.iloc[1], 5)
        # T0 = 0.003333
        # T1 = round((N/tl)*T0, 5)
        # _logger.debug('T0: {}  |  T1: {}'.format(T0, T1))
        # df.index = (pd.date_range(0, periods=N, freq='{0:.2f}ms'.format(T0*1e3)))
        # rescaled_df = df.resample('{0:.2f}ms'.format(T1*1e3)).pad()
    
        # Handle data as numpy array
        # num_df = rescaled_df[['state_i', 'action']].to_numpy(dtype='int')
        # num_df = df[['state_i', 'action']].to_numpy(dtype='int')
        num_df = df[['state_i', 'action']].values

        # States
        s = num_df[:, 0]
        s = s.reshape(-1, 1)
        _logger.debug('States: \n{}'.format(s))

        # Actions
        a = num_df[:, 1]# - 1
        a = a.reshape(-1, 1)
        _logger.debug('Actions: \n{}'.format(a))

        # Make s and a into columns and concatenate them
        traj = np.concatenate((s.reshape(-1, 1), a), axis=1)
        trajs.append(traj)

    # Trajectory list into numpy 3D array
    trajs = np.stack(trajs)

    return trajs

def send_ssl_email(user, pwd, recipient, subject, body):
    import smtplib

    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server_ssl.ehlo()
        # server.starttls()
        server_ssl.login(user, pwd)
        server_ssl.sendmail(FROM, TO, message)
        server_ssl.close()
        _logger.info('Sent')
    except:
        _logger.info('Failed to send email')


def main(args):
    args = parse_args(args)
    # setup_logging(args.loglevel)
    
    # Load and parse json configuration
    config_file = os.path.join(os.getcwd(), 'config.json')
    print(config_file)
    with open(config_file) as f:
            cfg = json.load(f)
    
    grid = cfg['grid']
    grid_axes = cfg['grid_axes']
    discount = cfg['discount']
    epochs = cfg['epochs']
    learning_rate = cfg['learning_rate']
    traj_len = cfg['traj_len']

    # Path where all output data will be stored
    out_dir = 'Maxent_ep{0}_tl{1}_{2}'.format(epochs, traj_len, tstamp())

    if args.send_email:

        mail_sender = 'aranea.notifications01@gmail.com'
        pwd = getpass('Password for {}: '.format(mail_sender))
        mail_recipient = 'hernandez.c.aa@m.titech.ac.jp'
        subject = 'MaxEnt IRL calculation finished'
        

    _logger.info(
        "Starting MaxEnt IRL calculations with:\n {}\nSave to csv: {}, Draw plots: {}".format(json.dumps(cfg, sort_keys=True, indent=4), bool(args.save_csv), bool(args.save_plots)))

    # Read csv logs from data_dir into trajectories
    csvs = [i for i in glob.glob(os.path.join(args.data_dir, '*.csv'))]
    trajectories = read_trajectories(csvs, traj_len)

    
    # Load the state transition probabilities
    tp = np.load(os.path.join(args.data_dir, 'trans_prob.npy'))

    # Construct a mothworld object
    mw = neo_mothworld.Mothworld(grid, grid_axes, discount, tp)
    # mw = gridworld.Gridworld(grid_size=16, wind=0.3, discount=0.9)

    # Initialize the feature matrix
    # feature_matrix = mw.feature_matrix(feature_map="coord")
    feature_matrix = mw.feature_matrix()
    _logger.info('Shape of feature matrix: {}'.format(feature_matrix.shape))
    _logger.debug('Feature matrix{0}:\n{1}\n'.format(
        feature_matrix.shape, feature_matrix))

    # Extract a reward function using MaxEnt IRL and the moth trajectories
    _logger.info('Calculating reward function')
    r, grad_sums = maxent.irl(feature_matrix, mw.n_actions, discount,
        mw.transition_probability, trajectories, epochs, learning_rate)
    
    _logger.info('Losses: {}'.format(grad_sums))
    _logger.info('Mean loss: {}'.format(grad_sums.mean()))
    # _logger.info('Transition probability {}:\n{}'.format(mw.transition_probability.shape, mw.transition_probability[0][0]))
    # for i in range(mw.transition_probability.shape[0]):
        # _logger.info('Sum of s{}: {}'.format(i, np.sum(mw.transition_probability[i])))
    # print('Shape of reward: {}'.format(r.shape))

    # Store extracted Q value
    _logger.info('Calculating policy')
    moth_policy = value_iteration.find_policy(mw.n_states, mw.n_actions,
                                         mw.transition_probability, r, 
                                         discount)

    moth_policy = moth_policy.T
    _logger.debug('Policy: {}\n{}'.format(moth_policy.shape, moth_policy))
    # action_labels = ['Surge', 'Turn_cw', 'Turn_ccw']
    action_labels = ['Surge', 'Hard CW', 'Soft CW', 'Hard CCW', 'Soft CCW']

    # v = value_iteration.optimal_value(mw.n_states, mw.n_actions, mw.transition_probability, [r[s] for s in range(mw.n_states)], discount)
    # print('Action value: {}\n{}'.format(v.shape, v))

    ex_reward = r.reshape(*grid_axes)   # extracted reward
    
    if args.save_csv:
      
      Path(os.path.join(args.data_dir, out_dir)).mkdir(
          parents=True, exist_ok=True)
      tmp_dir = os.path.join(args.data_dir, out_dir)
      
      _logger.info('Writing extracted reward to csv')
      pd.DataFrame(ex_reward).to_csv(os.path.join(tmp_dir, 'Reward.csv'))
      
      _logger.info('Writing extracted action-value to csv')
      for i in range(len(moth_policy)):
        pd.DataFrame(moth_policy[i].reshape(grid_axes)).to_csv(
            os.path.join(tmp_dir, 'policy_{0}.csv'.format(action_labels[i])))

    if args.save_plots:

        Path(os.path.join(args.data_dir, out_dir)).mkdir(
          parents=True, exist_ok=True)
        tmp_dir = os.path.join(args.data_dir, out_dir)

        sns.set_context('paper')

        _logger.info('Plotting reward')
        # fig, ax_r = plt.subplots(2, 1, figsize=(8,6))
        fig, ax_r = plt.subplots()
        ax_r = sns.heatmap(ex_reward.T, center=0,
                           yticklabels=cfg["substate_ticks"]["antennae"])
        ax_r.set_xlabel(cfg["state_labels"]["lv1"])
        ax_r.set_ylabel('{}'.format(cfg["state_labels"]["lv2"]))
        ax_r.invert_yaxis()
        
        if cfg["draw_subgrid"]:
            ax_r.hlines(cfg["subgrid_ticks"], *ax_r.get_ylim(), linestyle='--', linewidth=0.5, color='w')

        # df_r = pd.DataFrame(columns=['None', 'Right', 'Left', 'Both'], data=ex_reward)
        # sns.lineplot(data=df_r, ax=ax_r[1])
        # ax_r.set_title('Extracted reward')
        
        fig.suptitle('Epochs: {}, Trajectory length: {}, Learning rate: {}'.format(
            cfg['epochs'], cfg['traj_len'], cfg['learning_rate']))
        
        plt.savefig(os.path.join(tmp_dir, 'Reward.png'), dpi=300)

        _logger.info('Plotting policy')
        fig, ax_q = plt.subplots(1, mw.n_actions, figsize=(12,6.8), sharey=True)
        # ax_q = ax_q.ravel()
        # fig.tight_layout(rect=[0.01, 0, .9, 1])
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        
        # for i in range(len(moth_policy)):
        for i, ax in enumerate(ax_q.flat):

            # df_pi = pd.DataFrame(columns=['None', 'Right', 'Left', 'Both'], data=moth_policy[i].reshape(*grid_axes))
            # sns.lineplot(data=df_pi, ax=ax_q[i])
          
            sns.heatmap(moth_policy[i].reshape(*grid_axes), center=0,
            ax=ax,
            cbar=i == 0,
            vmin=0, vmax=1,
            cbar_ax=None if i else cbar_ax)
            
            ax.set_title('{}'.format(action_labels[i]))
            ax.invert_yaxis()
            ax.set_xlabel('{}'.format(cfg["state_labels"]["lv2"]))

            if cfg["draw_subgrid"]:
                ax.vlines(cfg["subgrid_ticks"], *ax.get_ylim(),
                        linestyle='--', linewidth=0.5, color='w')
            
            ax_q[0].set_ylabel(cfg["state_labels"]["lv1"])
        
        fig.suptitle('Epochs: {}, Trajectory length: {}'.format(
            cfg['epochs'], cfg['traj_len']))
        plt.savefig(os.path.join(tmp_dir, 'Policy.png'), dpi=300)
        # plt.show()
        
    
    if args.send_email:

        _logger.info('Sending email notification to: {}'.format(mail_recipient))

        body = 'Configuration parameters:\n{}\nSave to csv: {}, Draw plots: {}'.format(json.dumps(cfg, sort_keys=True, indent=4), bool(args.save_csv), bool(args.save_plots))
        send_ssl_email(mail_sender, pwd, mail_recipient, subject, body)

    _logger.info("Script ends here")

if __name__ == "__main__":
    main(sys.argv[1:])