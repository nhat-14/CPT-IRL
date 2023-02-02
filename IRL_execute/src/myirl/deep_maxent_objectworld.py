#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file estimates the reward function of a silkmoth based on Deep Maximum Entropy IRL and trajectories obtained from wind tunnel experiments.
"""

from myirl.irl.mdp import neo_mothworld
from myirl.irl.mdp import mothworld
from myirl.irl.mdp import gridworld
from myirl.irl.mdp import objectworld
from myirl.irl import value_iteration
from myirl.irl import maxent
from myirl.irl import deep_maxent
from myirl import __version__
from getpass import getpass
from pathlib import Path
from tqdm import tqdm
import json
import glob
from os.path import isfile, join
import os
import argparse
import sys
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="sans-serif", rc={"font.sans-serif": ["FreeSans", "Arial"]})

__author__ = "Cesar Hernandez"
__copyright__ = "Cesar Hernandez"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def main(args):
    """
    Run deep maximum entropy inverse reinforcement learning on the objectworld
    MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    structure: Neural network structure. Tuple of hidden layer dimensions, e.g.,
        () is no neural network (linear maximum entropy) and (3, 4) is two
        hidden layers with dimensions 3 and 4.
    """

    print("Starting Deep MaxEnt IRL calculations")

    wind = 0.3
    trajectory_length = 1000
    l1 = l2 = 0

    grid_size = 10
    discount = 0.9
    n_objects = 15
    n_colours = 2
    n_trajectories = 20
    epochs = 200
    learning_rate = 0.01
    structure = (3,3)

    ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,
                                 discount)
    ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
    policy = value_iteration.find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                         ground_r, ow.discount, stochastic=False)
    trajectories = ow.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: policy[s])
    feature_matrix = ow.feature_matrix(discrete=False)
    print(feature_matrix.shape)
    r = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
                        ow.n_actions, discount, ow.transition_probability, trajectories, epochs,
                        learning_rate, l1=l1, l2=l2)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()
    # args = parse_args(args)
    # setup_logging(args.loglevel)

    print("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
