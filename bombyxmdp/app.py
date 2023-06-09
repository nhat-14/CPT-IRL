"""app.py: Generate state-action trajectories plus other useful stats from Shigaki's 
2020 tethered moth experiments which incorporate wind stimuli.
IN: path of directory containing log files (csv format)
OUT: Csv files with state-action trajectories"""

__author__      = "Duc-Nhat Luong"
__copyright__   = "Copyright 2022, The CPT-IRL Project"
__credits__     = ["Duc-Nhat Luong, Cesar Hernandez-Reyes"]
__license__     = "MIT"
__version__     = "2.0.0"
__maintainer__  = "Duc-Nhat Luong"
__email__       = "nhat.luongduc@gmail.com"
__status__      = "Production"


import os
from datetime import datetime
from os.path import join
import numpy as np
import pandas as pd
import preprocessing
import markov_decission_process as mdp
import config


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
    numeric_states = config.numeric_states
    categoric_states = config.categoric_states
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

    pd.set_option('display.max_rows', len(mdp_demos))
    print(mdp_demos['action'].value_counts())
    pd.reset_option('display.max_rows')

    out_dir = create_output_folder()
    np.save((join(out_dir, 'trans_prob.npy')), transition_prob)
    # export_csv(mdp_edges, 'bin_edges.csv', out_dir)
    export_csv(mdp_demos, 'fmdp_demos.csv', out_dir)
    for i, g in mdp_demos.groupby((mdp_demos.Time.diff() < 0).cumsum()):
        export_csv(g, f'{len(g.index)}-{i+1}.csv', out_dir)