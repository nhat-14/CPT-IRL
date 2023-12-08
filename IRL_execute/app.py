"""
This file estimates the reward function of a silkmoth based on Deep 
Maximum Entropy IRL and trajectories obtained from wind tunnel experiments.
"""

import os
from pathlib import Path
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from IRL_execute.decision_tree_regression import DecisionTreeRegressor
import utilities as util

sns.set(font="sans-serif", rc={"font.sans-serif": ["FreeSans", "Arial"]})


def read_trajectories(directory):
    """
    Make a (T, L, 2) 3D numpy array where T is the number 
    of csv files and L is the trajectory length.
    2 means two columns (of states-actions)
    """
    csvs = util.get_csv_files(directory, '[!f, !b]*.csv')
    trajs = []
    for csv_file in csvs:
        file_name = os.path.basename(csv_file)
        num_rows = int(file_name.split('-')[0])
        rows_read = 0

        # Split a long trial into several short trials
        while (num_rows // cfg.traj_len) > 0: 
            df = pd.read_csv(csv_file, skiprows=range(1, rows_read), nrows=cfg.traj_len)

            # Handle data as numpy array
            num_df = df[["state_i", "action"]].values

            # States (column vector with 1 column)
            states = num_df[:, 0]
            states = states.reshape(-1, 1)

            # Actions (column vector with 1 column)
            actions = num_df[:, 1]
            actions = actions.reshape(-1, 1)

            # Join state and action arrays along horizontal axis as columns.
            traj = np.concatenate((states, actions), axis=1)
            trajs.append(traj)
            num_rows -= cfg.traj_len
            rows_read += cfg.traj_len
    # List of trajectories -> 3D numpy array
    trajs = np.stack(trajs)
    return trajs


def plot_reward_function(data, save_path=None, annot=True):
    """Plot reward function for each state
    Args:
            df (pandas.DataFrame): Data from csv
    """
    plot_title = f'Epochs: {cfg.epochs}, Trajectory length: {cfg.traj_len}, \
        Discount: {cfg.discount}, Learning rate: {cfg.learning_rate}'
    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax = plt.subplots(figsize=(7, 3))
    ax = sns.heatmap(
        data,
        # center=0,
        cmap='magma',
        yticklabels=cfg.substate_ticks[cfg.state_names[1]],
        annot=annot,
        cbar_kws={'label': 'Reward'})
    ax.set_xlabel('Blank duration bins')
    ax.set_ylabel('Antennae')
    ax.set_title(plot_title)
    fig.tight_layout()
    plt.savefig(join(save_path, 'Reward.png'), dpi=300)
    plt.show()


def plot_policy(out_dir):
    tmp_dir = join(cfg.input_dir, out_dir)
    plot_title = f'Epochs: {cfg.epochs}, Trajectory length: {cfg.trajectory_len}, \
        Discount: {cfg.discount}, Learning rate: {cfg.learning_rate}'

    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax_q = plt.subplots(
        1, mothworld.n_actions, figsize=(12, 6.8), sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, ax in enumerate(ax_q.flat):

        sns.heatmap(moth_policy[i].reshape(*cfg.n_sub_states), center=0,
                    ax=ax,
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax,
                    xticklabels=cfg.substate_ticks[cfg.state_names[1]], annot=True)

        ax.set_title(f'{cfg.action_labels[i]}')
        ax.invert_yaxis()
        ax.set_xlabel(f'{cfg.state_labels[cfg.state_names[1]]}')

        if cfg.draw_subgrid:
            ax.vlines(cfg.subgrid_ticks, *ax.get_ylim(),
                        linestyle='--', linewidth=0.5, color='w')

        ax_q[0].set_ylabel(cfg.state_labels[cfg.state_names[0]])

    fig.suptitle(plot_title)
    plt.savefig(join(tmp_dir, 'Policy.png'), dpi=300)
    print('Plots saved to disk')


def create_output_folder():
    """
    Create an output folder to store learning results
    """ 
    time_log = util.get_timestamp()
    config_log = f'e{cfg.epochs}_t{cfg.traj_len}_g{cfg.discount}'
    dir_name = f'DeepMaxEnt_{config_log}_{time_log}'
    path_obj = Path(join(cfg.input_dir, dir_name))
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
  

def assign_reward_along_trajectory(lookup, mdp_demos):
    """
    Having the reward function, we use it to assign the
    state reward in each step of the moth demostration
    """
    mdp_demos['reward'] = mdp_demos.state_i.map(lookup.set_index('state').reward)
    return mdp_demos


def feature_fitting(reward_matrix, full_demos_data):
    """
    Using decision tree regression to auto select the suitable feature vector
    """
    reward_table = pd.DataFrame({'reward': reward_matrix[:]})
    reward_table['state'] = reward_table.index

    full_demos_data['reward'] = full_demos_data.state_i.map(reward_table.set_index('state').reward)
    feature_name_list = cfg.feature_pool + ["reward"]
    small_data = full_demos_data[feature_name_list]
    data_len = 10000
    X = small_data.iloc[:data_len, :-1].values
    Y = small_data.iloc[:data_len, -1].values.reshape(-1,1)
    regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
    regressor.fit(X,Y)
    regressor.print_tree()
    regressor.list_conditional_node()
    print(regressor.get_node_list())
    
    next_feature_set = []
    for i, name in enumerate(feature_name_list):
        for j in regressor.get_node_list():
            if j == i:
                next_feature_set.append(name)
    print(next_feature_set)
    next_matrix = full_demos_data.groupby('state_i')[next_feature_set].median()
    next_matrix = state_lacking_handel(next_matrix, next_feature_set)
    # return result.astype('uint8')
    return next_matrix


def state_lacking_handel(lacking_matrix, column_names):
    # fill 0 as the features values of missing states 
    column_values = [0] * len(column_names)
    existed_states = list(lacking_matrix.index.values)
    for i in range(cfg.n_states):
        if i not in existed_states:
            lacking_matrix.loc[i] = column_values
    return lacking_matrix


def get_feature_matrix(full_demos_data):
    """
    Get feature dataframe from the concated demos data from the moth generated 
    from data processing step in bombyxmdp
    """
    matrix = np.eye(cfg.n_states)   # unit vector feature
    if cfg.feature_fitting_loop == 0:
        features_name = ['wind', 'angular_vel']
        # features_name = ['whiff', 'hit_rate', 'lasthit', 'region_x', 'obstacle_distance']
        # mean() or median() is better?
        matrix = full_demos_data.groupby('state_i')[features_name].median()
        # matrix = matrix.astype('uint8')
        # matrix['wind'] = matrix['wind'].astype('uint8')
        # features['angular_vel'] = np.sign(features.angular_vel).astype('int')
        matrix = state_lacking_handel(matrix, features_name)
    return matrix