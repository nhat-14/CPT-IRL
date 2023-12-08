"""
This file contains fixed configurations for processing the data
"""
import os

#============ Environment shape and dimension (in mm) ============#
environment = "free"   # {free, rectangle, cylinder}
xlim = (0,600)
ylim = (-360,360)
GOAL_RADII  = 50.5          # Distance of agent to source to consider success

if environment == "free":
    INPUT_DIR = os.path.join(os.getcwd(), "data") 
elif environment == "rectangle":
    INPUT_DIR = os.path.join(os.getcwd(), "data_rectangle") 

# Rectangle obstacle details
width           = 15
length          = 150
lower_corner    = (200,-75)

# Time related parameters 
exp_timeout = 260       # Maximum allowed runtime of a trial from VR system
time_step   = 0.033333333 # Time interval between each recorded data
WINDOW_SIZE = 1         # Moving average window size: 15 := 0.5s

# State and Action encoding parameters
# [name, bins, skewed, use logscale, use_kmeans]
numeric_states      = ['log_tblank', 16, True, True, True]
categoric_states    = ['antennae'] # or ['antennae', 'wind']
# categoric_states    = ['antennae', 'region_x'] # or ['antennae', 'wind']


#================================= IRL Config ================================

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
                    # "antennae": ["None_A", "Right_A", "Left_A", "Both_A", "None_B", "Right_B", "Left_B", "Both_B"], 
                    "antennae": ["None", "Right", "Left", "Both"], 
                    "wind": ["Left", "Front", "Right", "Back"]
                }

# learning related configurations
discount        = 0.9       # Discount factor
epochs          = 200       # Training epochs for the gradient descent
learning_rate   = 0.01
traj_len        = 1300      # Trajectory length
NNstructure     = (4, 3)    # Neural network structure tuple

# Action related configurations
n_actions           = 4
action_labels       = ["stop", "surge", "turn_ccw", "turn_cw"]
actions             = [0, 1, 2, 3]

# Conditions configurations
draw_subgrid = False
l2reg = False
subgrid_ticks = [4, 8, 12]

# feature_pool = ['wind', 'linear_vel', 'angular_vel', "whiff", "hit_rate", "lasthit", 
#                 "region_x", "region_y", "obstacle_distance", "heading_obs", "antennae", "tblank"]
feature_pool = ['wind', 'linear_vel', 'angular_vel', "whiff", "hit_rate", "lasthit", "antennae", "tblank"]
feature_fitting_loop = 0