"""
Class that represents a silkmoth as an RL agent
"""
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
        return 'x, y, theta'

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
        self.pos.x += dt * rdot * np.cos(theta + dt * omega / 2)
        self.pos.y += dt * rdot * np.sin(theta + dt * omega / 2)
        self.theta += dt * omega
        if(self.theta >= 2 * np.pi): theta -= 2 * np .pi
        elif(self.theta < 0): theta += 2 * np .pi