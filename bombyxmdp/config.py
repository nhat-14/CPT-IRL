"""
This file contains fixed configurations for processing the data
"""
import os

basepath    = os.getcwd()
INPUT_DIR   = os.path.join(basepath, "data") # folder has data from VR system
GOAL_RADII  = 50.5 # Distance of agent to source to consider source reached
WINDOW_SIZE = 1    # Moving average window size: 15 == 0.5s

# name, bins, skewed, use logscale, use_kmeans
numeric_states      = ['log_tblank', 16, True, True, True]
categoric_states    = ['antennae'] # or ['antennae', 'wind']

# Environment dimensions (in mm)
xlim = (0,600)
ylim = (-360,360)