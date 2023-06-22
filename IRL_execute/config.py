import os
from os.path import join

# path file system related configurations
basepath = os.getcwd()
input_dir = join(basepath, "data")

# states related configurations
# n_sub_states    = [16, 4]
n_sub_states    = [4, 16]
n_states        = n_sub_states[0]*n_sub_states[1]

state_names     = ["tblank", "antennae"]
state_labels    = {
                    "hits": "Cumulative hits",
                    "tblank": "Blank duration",
                    "antennae": "Antennae input",
                    "wind": "Wind input direction"
                }
substate_ticks  = {
                    "antennae": ["None", "Right", "Left", "Both"], 
                    "wind": ["Left", "Front", "Right", "Back"]
                }

# learning related configurations
discount            = 0.9       # Discount factor
epochs              = 200       # Training epochs for the gradient descent
learning_rate       = 0.01
trajectory_len      = 1400  
NNstructure         = (4, 3)    # Neural network structure tuple

# Action related configurations
n_actions           = 4
action_labels       = ["stop", "surge", "turn_ccw", "turn_cw"]
actions             = [0, 1, 2, 3]

# Conditions configurations
draw_subgrid = False
l2reg = False
subgrid_ticks = [4, 8, 12]

auto_select_feature = False
fitting_loop = 10