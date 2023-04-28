"""
This file contains fixed configurations for processing the data
"""
import os

basepath = os.getcwd()
INPUT_DIR = os.path.join(basepath, "data") # folder has data from VR system
GOAL_RADII = 50.0 # Distance of agent to source to consider source reached