# -*- coding: utf-8 -*-
"""
Class that represents a silkmoth as an RL agent
"""
import pandas as pd
import numpy as np
from utils.geometry import Point

class SilkMoth(object):
    def __init__(self, x0, y0, th0):

        self.pos = Point(x0, y0)
        self.theta = np.deg2rad(th0)
        # IAA: inter antennal angle in degrees
        self.iaa = np.deg2rad(30)
        # Antennae length in mm
        self.antenna_length = 5

    def __str__(self):
        return 'x:{:.2f}, y:{:.2f}, th:{:.2f}'.format(self.pos.x, self.pos.y, self.theta)

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

    def move(self, rdot, omega, dt):
        """Move based on linear and angular velocity commands

        Args:
            rdot (float): Linear velocity
            omega (float): Angular velocity
            dt (float): Time differential in seconds
        """
        r = self.pos
        theta = self.theta

        pre_x=self.pos.x
        pre_y=self.pos.y
        pre_theta=self.theta
        self.pos.x += dt * rdot * np.cos(theta + dt * omega / 2)
        self.pos.y += dt * rdot * np.sin(theta + dt * omega / 2)
        self.theta += dt * omega
        rx=self.right_antenna.x
        ry=self.right_antenna.y
        lx=self.left_antenna.x
        ly=self.left_antenna.y

        if ((rx>=215) and (rx<=235) and (ry>=-100) and (ry<=100)) or ((lx>=215) and (lx<=235) and (ly>=-100) and (ly<=100)):
            self.pos.x=pre_x
            self.pos.y=pre_y
            self.theta=pre_theta
        if (rx>=500) or (rx<=0) or (ry>=200) or (ry<=-200) or (lx>=500) or (lx<=0) or (ly>=200) or (ly<=-200):
            self.pos.x=pre_x
            self.pos.y=pre_y
            self.theta=pre_theta
        if(self.theta >= 2 * np.pi): theta -= 2 * np .pi
        elif(self.theta < 0): theta += 2 * np .pi
