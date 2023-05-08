"""
This file estimates the reward function of a silkmoth based on Deep 
Maximum Entropy IRL and trajectories obtained from wind tunnel experiments.
"""

import glob
import os
from datetime import datetime
from pathlib import Path
from os.path import join

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import neo_mothworld
from myirl import value_iteration
from myirl import deep_maxent
import config as cfg

sns.set(font="sans-serif", rc={"font.sans-serif": ["FreeSans", "Arial"]})

def read_trajectories(traj_length=0):
    """
        Make a (T, L, 2) 3D numpy array where T is the number 
        of csv files and L is the trajectory length.
        2 means two columns (of states-actions)
    """
    csvs = load_csv_files()
    trajs = []
    for csv_file in csvs:
        print(f'Working on: {os.path.basename(os.path.splitext(csv_file)[0])}.csv')
        dataframe = pd.read_csv(csv_file, index_col=None, nrows=traj_length)

        # Handle data as numpy array
        num_df = dataframe[["state_i", "action"]].values

        # States (column vector with 1 column)
        states = num_df[:, 0]
        states = states.reshape(-1, 1)

        # Actions (column vector with 1 column)
        actions = num_df[:, 1]
        actions = actions.reshape(-1, 1)

        # Join state and action arrays along horizontal axis as columns.
        traj = np.concatenate((states, actions), axis=1)
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


def plot_policy(out_dir):
    tmp_dir = join(cfg.input_dir, out_dir)
    plot_title = f'Epochs: {cfg.epochs}, Trajectory length: {cfg.trajectory_len}, \
        Discount: {cfg.discount}, Learning rate: {cfg.learning_rate}'

    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax_r = plt.subplots(figsize=(7, 4))

    ax_r = sns.heatmap(ex_reward.T,
                        center=0,
                        cmap='magma',
                        yticklabels=cfg.substate_ticks[cfg.state_names[1]],
                        annot=True,
                        cbar_kws={'label': 'Reward'})

    ax_r.set_xlabel(cfg.state_labels[cfg.state_names[0]])
    ax_r.set_ylabel(cfg.state_labels[cfg.state_names[1]])
    ax_r.invert_yaxis()

    if cfg["draw_subgrid"]:
        ax_r.hlines(cfg["subgrid_ticks"], *ax_r.get_ylim(),
                    linestyle='--', linewidth=0.5, color='w')

    fig.suptitle(plot_title)

    plt.savefig(join(tmp_dir, 'Reward.png'), dpi=300)
    plt.show()

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


def save_csv(out_dir):
    tmp_dir = join(cfg.input_dir, out_dir)

    # Saving reward to csv
    pd.DataFrame(ex_reward).to_csv(join(tmp_dir, 'Reward.csv'))
    
    np.savetxt(join(tmp_dir,'simple_policy.csv'), simple_policy, delimiter=',', fmt='%d')
    np.savetxt(join(tmp_dir,'moth_policy.csv'), moth_policy, delimiter=',')

    # Saving policy to h5 and csv
    with h5py.File(join(tmp_dir, 'policy.h5'), 'w') as hf:
        hf.create_dataset("policy",  data=moth_policy.reshape(mothworld.n_actions, *cfg.n_sub_states))

    for i in range(len(moth_policy)):
        pd.DataFrame(moth_policy[i].reshape(cfg.n_sub_states)).to_csv(
            join(tmp_dir, 'policy_{0}.csv'.format(cfg.action_labels[i])))

    print('Reward and policy saved to disk')


def get_current_time():
    """
    Get the current time until seconds under the string format
    """
    return datetime.now().strftime('%m%d_%H%M%S')


def load_csv_files():
    """
    return a list of absolute path string. Each path point to a csv file
    (except the feature.csv) from input_dir containing trajectories data.
    """
    # Load csv logs except the feature.csv
    paths = get_file_path('[!f]*.csv')
    paths_with_full_name = glob.glob(paths)
    return paths_with_full_name


def get_file_path(filename):
    """
    Return the absolute path of a file in the data input folder give a name.
    """
    return join(cfg.input_dir, filename)


def generate_output_folder_name():
    """
    Folder where all output data will be stored. Name will contain time and 
    learning configurations
    """
    time_log = get_current_time()
    config_log = f'e{cfg.epochs}_t{cfg.trajectory_len}_g{cfg.discount}'
    return f'DeepMaxEnt_{config_log}_{time_log}'


def create_output_folder():
    """
    Create an output folder to store learning results
    """ 
    dir_name = generate_output_folder_name()
    path_obj = Path(join(cfg.input_dir, dir_name))
    path_obj.mkdir(parents=True, exist_ok=True)
    return dir_name


if __name__ == "__main__":
    # Load the state transition probabilities
    trans_prob = np.load(get_file_path('trans_prob.npy'))

    # Construct a mothworld object
    mothworld = neo_mothworld.MothWorld(cfg.n_states, cfg.n_sub_states, cfg.discount, trans_prob)

    # Initialize the feature matrix
    feature_path = get_file_path('features.csv')
    feature_matrix = pd.read_csv(feature_path).values
    print(f'Shape of feature matrix: {feature_matrix.shape}')

    # Extract a reward function using MaxEnt IRL and the moth trajectories
    structure = (4, 3)
    print(f'NN structure: {structure}; learning rate: {cfg.learning_rate}')

    # Whether to use L2 regularization in the gradient descent l2reg
    (l1, l2) = tuple(cfg.l2reg) if cfg.l2reg else (0,0)  

    # Calculating reward
    trajectories = read_trajectories(cfg.trajectory_len)
    r = deep_maxent.irl((feature_matrix.shape[1],) + structure,
                        feature_matrix,
                        mothworld.n_actions,
                        mothworld.discount,
                        mothworld.transition_probability,
                        trajectories,
                        cfg.epochs,
                        cfg.learning_rate,
                        l1=l1,
                        l2=l2)

    # Reshape reward
    ex_reward = r.reshape(*cfg.n_sub_states)
    mean_reward = [round(np.mean(ex_reward[:, i]), 2) for i in range(ex_reward.shape[1])]
    print(f'Mean reward: {mean_reward}')

    # Store extracted Q value and Calculating policy
    moth_policy = value_iteration.find_policy(mothworld.n_states, 
                                              mothworld.n_actions,
                                              mothworld.transition_probability, 
                                              r,
                                              cfg.discount, 
                                              threshold=1e-2)
    
    simple_policy = np.array([np.argmax(moth_policy[i,:]) for i in range(mothworld.n_states)])
    simple_policy = simple_policy.reshape(mothworld.substate).T

    # Save policy into csv
    pd.DataFrame(moth_policy).to_csv('raw_policy.csv', index=False)
    moth_policy = moth_policy.T
    print(f'Policy: {moth_policy.shape}\n{moth_policy}')

    output_folder = create_output_folder()
    save_csv(output_folder)
    # plot_policy(output_folder)