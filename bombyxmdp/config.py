"""
This file contains fixed configurations for processing the data
"""
import os

INPUT_DIR   = os.path.join(os.getcwd(), "data") # folder has data from VR system
GOAL_RADII  = 50.5 # Distance of agent to source to consider source reached
WINDOW_SIZE = 1    # Moving average window size: 15 == 0.5s

# name, bins, skewed, use logscale, use_kmeans
numeric_states      = ['log_tblank', 16, True, True, True]
categoric_states    = ['antennae'] # or ['antennae', 'wind']


# Maximum runtime of a trial from VR system to be considered
exp_timeout = 260   

# Environment dimensions (in mm)
xlim = (0,600)
ylim = (-360,360)

# Time interval between each recorded data
time_step = 0.0333333333333333

# Obstacle details
width = 15
length = 150
lower_corner = (200,-75)