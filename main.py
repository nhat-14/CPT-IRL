"""
Generate state-action trajectories plus other useful stats from Shigaki's 
2020 tethered moth experiments which incorporate wind stimuli.
IN: path of directory containing log files (csv format)
OUT: Csv files with state-action trajectories
"""

__author__      = "Duc-Nhat Luong"
__copyright__   = "Copyright 2022, The CPT-IRL Project"
__credits__     = ["Duc-Nhat Luong, Cesar Hernandez-Reyes"]
__license__     = "MIT"
__version__     = "2.0.0"
__maintainer__  = "Duc-Nhat Luong"
__email__       = "nhat.luongduc@gmail.com"
__status__      = "Production"

import os
import numpy as np
import pandas as pd
import h5py

import config
import utilities as util
from bombyxmdp import preprocessing
from IRL_execute import app
from IRL_execute import neo_mothworld
from IRL_execute import value_iteration
from IRL_execute import deep_maxent

######################### Data Preprocessing #############################
# 1) Extracting features from raw data
# 2) Encoding states and actions
# 3) Export the data and save them on the disk
##########################################################################

# Processing raw data to features for machine learning
dfs = preprocessing.merge_data()

# Encode state, actions and obtain state-action probabilities
mdp_demos, mdp_edges, transition_prob = util.get_expert_demos(dfs.copy())    

# Export data for learning process
out_dir = util.create_output_folder(f'{config.environment}_{util.get_timestamp()}')
np.save((os.path.join(out_dir, 'trans_prob.npy')), transition_prob)
util.export_csv(mdp_edges, 'bin_edges.csv', out_dir)
util.export_csv(mdp_demos, 'fmdp_demos.csv', out_dir)
for i, g in mdp_demos.groupby((mdp_demos.Time.diff() < 0).cumsum()):
    util.export_csv(g, f'{len(g.index)}-{i+1}.csv', out_dir)


#################### Inverse Reinforcement Learning ######################
# 1) Learning to get a reward function
# 2) Encoding states and actions
##########################################################################
trajectories = app.read_trajectories(directory=out_dir)
feature_matrix = app.get_feature_matrix(mdp_demos)
mothworld = neo_mothworld.MothWorld(transition_prob)    # a mothworld object

interation = config.feature_fitting_loop + 1
for i in range(interation):
    print(f"===Feature fitting interation {i}====")
    NeuronNet_structure = (feature_matrix.shape[1],) + config.NNstructure

    # Calculating reward
    r = deep_maxent.irl(
        NeuronNet_structure,
        feature_matrix,
        mothworld.n_actions,
        mothworld.discount,
        mothworld.transition_probability,
        trajectories,
        config.epochs,
        config.learning_rate,
        l1=0,
        l2=0)
    
    # update reward function
    if config.feature_fitting_loop > 0:
        feature_matrix = app.feature_fitting(r, mdp_demos)
#===========================================================================

# Reshape reward
reward = r.reshape(config.n_sub_states)

# Store extracted Q value and Calculating policy
moth_policy = value_iteration.find_policy(mothworld.n_states,
                                            mothworld.n_actions,
                                            mothworld.transition_probability,
                                            r,
                                            config.discount,
                                            threshold=1e-2)

# pd.DataFrame(moth_policy).to_csv('raw_policy.csv', index=False)
util.export_csv(pd.DataFrame(moth_policy), 'raw_policy.csv', out_dir)
simple_policy = np.array([np.argmax(moth_policy[i,:]) for i in range(config.n_states)])
simple_policy = simple_policy.reshape(config.n_sub_states)

moth_policy = moth_policy.T

util.export_csv(pd.DataFrame(reward), 'Reward.csv', out_dir)
np.savetxt(os.path.join(out_dir,'simple_policy.csv'), simple_policy, delimiter=',', fmt='%d')
np.savetxt(os.path.join(out_dir,'moth_policy.csv'), moth_policy, delimiter=',')

# Saving policy to h5 and csv
with h5py.File(os.path.join(out_dir, 'policy.h5'), 'w') as hf:
    hf.create_dataset("policy",  data=moth_policy.reshape(mothworld.n_actions, *config.n_sub_states))

for i in range(len(moth_policy)):
    pd.DataFrame(moth_policy[i].reshape(config.n_sub_states)).to_csv(
        os.path.join(out_dir, f'policy_{config.action_labels[i]}.csv'))

print('Reward and policy saved to disk')
# app.plot_policy(out_dir)
app.plot_reward_function(reward, out_dir)