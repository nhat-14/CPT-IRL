# -*- coding: utf-8 -*-
"""
Class that represents a silkmoth as an RL agent
"""
import pandas as pd
import numpy as np
from utils.geometry import Point
import controllers.infotaxis_core as core


class ItxAgent(object):
    def __init__(self, x0, y0):

        self.pos = Point(x0, y0)
        self.theta = np.pi
        # IAA: inter antennal angle in degrees
        self.iaa = np.deg2rad(30)
        # Antennae length in mm
        # self.antenna_length = 5
        self.antenna_length = core.AGT_SIZE * 1e3

    def __str__(self):
        return 'x:{:.2f}, y:{:.2f}, th:{:.2f}'.format(self.pos.x, self.pos.y,
                                                      self.theta)

    @property
    def log_header(self):
        return 'x_mm,y_mm,theta_rad'

    @property
    def log_step(self):
        return [self.pos.x, self.pos.y, self.theta]

    @property
    def right_antenna(self):

        x = self.pos.x + self.antenna_length * np.cos(self.theta - self.iaa)
        y = self.pos.y + self.antenna_length * np.sin(self.theta - self.iaa)

        return Point(x, y)

    @property
    def left_antenna(self):

        x = self.pos.x + self.antenna_length * np.cos(self.theta + self.iaa)
        y = self.pos.y + self.antenna_length * np.sin(self.theta + self.iaa)

        return Point(x, y)

    def move(self, u):
        """Move based on linear and angular velocity commands

        Args:
            rdot (float): Linear velocity
            omega (float): Angular velocity
            dt (float): Time differential in seconds
        """
        # self.pos.x = u[0]
        self.pos.x = (u[0] * 1000)
        # self.pos.y = (u[1] * 1000)
        self.pos.y = ((u[1] - .36) * 1000)
        # self.pos.y = u[1]