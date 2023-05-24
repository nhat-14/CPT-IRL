"""
Generate state-action trajectories plus other useful stats from Shigaki's 
2020 tethered moth experiments which incorporate wind stimuli.
IN: path of directory containing log files (csv format)
OUT: Csv files with state-action trajectories
"""

import os
import datetime
from os.path import join
import numpy as np
import pandas as pd
import preprocessing
import markov_decission_process as mdp


def get_timestamp():
    """
    Return current time in date and time
    """
    tstamp = datetime.datetime.now()
    return str(tstamp.strftime('%m%d_%H%M%S'))


def get_tblank_bin_edges(mdp_space):
    """
    Get a list of time moment which acts as edges of bins
    for discretizing the timeline
    """
    edges = []
    for k, v in mdp_space.digi_edges.items():
        edges.append((k, pd.Series(v)))
    return pd.DataFrame(dict(edges))


def get_expert_demos(df):
    """
    Encoding the states and actions into discrete space
    """
    numeric_states = ['log_tblank', 16, True, True, True]
    categoric_states = ['antennae'] # or ['antennae', 'wind']
    mdp_space = mdp.MothMDP(df, numeric_states, categoric_states)
    mdp_space.encode_states()
    mdp_space.encode_actions()
    trans_prob = mdp_space.get_transition_probability()
    mdp_edges = get_tblank_bin_edges(mdp_space)
    mdp_demos = mdp_space.df.copy()
    return mdp_demos, mdp_edges, trans_prob


def create_output_folder():
    """
    Create output folder to save the proccessed data from original VRmoth data
    The folder name will have timestamp to distinguish learning attempts
    """
    basepath = os.getcwd()
    directory = join(basepath, f'rldemos_{get_timestamp()}')
    os.mkdir(directory)
    return directory


def export_csv(data, file_name, destination_folder):
    """
    Export processed data into csv files for learning using IRL_execute
    """
    filepath = join(destination_folder, file_name)
    data.to_csv(filepath, index=False)


if __name__ == "__main__":
    dfs = preprocessing.merge_data(timeout=260)
    mdp_demos, mdp_edges, transition_prob = get_expert_demos(dfs.copy())
    
    # ['wind', 'hits', 'linear_vel', 'angular_vel', 'log_twhiff', 'lasthit']].mean()
    # features = mdp_demos.groupby('state_i')[['wind', 'angular_vel']].median()
    features = mdp_demos.groupby('state_i')[['tblank', 'linear_vel', "antennae"]].median()
    # features['wind'] = features.wind.astype('uint8')
    features['tblank'] = features.tblank.astype('uint8')
    features['antennae'] = features.antennae.astype('uint8')
    features['linear_vel'] = features.linear_vel.astype('uint8')
    # features['angular_vel'] = np.sign(features.angular_vel).astype('int')

    # export all the results (bins, features, transittion matrix) into csv files
    out_dir = create_output_folder()
    np.save((join(out_dir, 'trans_prob.npy')), transition_prob)
    export_csv(mdp_edges, 'bin_edges.csv', out_dir)
    export_csv(features, 'features.csv', out_dir)
    export_csv(mdp_demos, 'mdp_demos.csv', out_dir)
    for i, g in mdp_demos.groupby((mdp_demos.Time.diff() < 0).cumsum()):
        export_csv(g, f'{len(g.index)}-{i+1}.csv', out_dir)