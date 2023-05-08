import os
from os.path import join

basepath = os.getcwd()
input_dir = join(basepath, "data")

n_states        = 64
n_sub_states    = [16, 4]
n_actions       = 4
discount        = 0.9       # Discount factor
epochs          = 200       # Training epochs for the gradient descent
learning_rate   = 0.01
trajectory_len = 109
action_labels = ["stop", "surge", "turn_ccw", "turn_cw"]
draw_subgrid = False
l2reg = False
subgrid_ticks = [4, 8, 12]
state_names = ["tblank", "antennae"]
state_labels = {
    "hits": "Cumulative hits",
    "tblank": "Blank duration",
    "antennae": "Antennae input",
    "wind": "Wind input direction"
}
substate_ticks = {
    "antennae": ["None", "Right", "Left", "Both"], 
    "wind": ["Left", "Front", "Right", "Back"]
}
