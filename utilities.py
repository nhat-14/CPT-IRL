import os
import glob
from datetime import datetime

import pandas as pd
import numpy as np

import bombyxmdp.markov_decission_process as mdp
import config as cfg


def get_csv_files(folder, file_name)-> list[str]:
    """
    Get csv files of trajectories obtained from Moth VR experiments
    """
    abs_file_name = os.path.join(folder, file_name)
    return list(glob.glob(abs_file_name))


def get_timestamp():
    """
    Return current time in date and time
    """
    tstamp = datetime.now()
    return tstamp.strftime('%m%d_%H%M%S')


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
    # name, bins, skewed, use logscale, use_kmeans
    numeric_states = cfg.numeric_states
    categoric_states = cfg.categoric_states
    mdp_space = mdp.MothMDP(df, numeric_states, categoric_states)
    mdp_space.encode_states()
    mdp_space.encode_actions()
    trans_prob = mdp_space.get_transition_probability()
    mdp_edges = get_tblank_bin_edges(mdp_space)
    mdp_demos = mdp_space.df.copy()
    return mdp_demos, mdp_edges, trans_prob


def create_output_folder(folder_name):
    """
    Create output folder to save the proccessed data from original VRmoth data
    The folder name will have timestamp to distinguish learning attempts
    """
    basepath = os.getcwd()
    directory = os.path.join(basepath, 'output', folder_name)
    os.mkdir(directory)
    return directory


def export_csv(data, file_name, destination_folder):
    """
    Export processed data into csv files for learning using IRL_execute
    """
    filepath = os.path.join(destination_folder, file_name)
    data.to_csv(filepath, index=False, float_format='%.2f')


def wrapping_angle_from_0to2pi(angle):
    mask = angle > np.pi
    angle[mask] = angle[mask] - 2*np.pi
    mask = angle < -np.pi
    angle[mask] = angle[mask] + 2*np.pi
    return angle