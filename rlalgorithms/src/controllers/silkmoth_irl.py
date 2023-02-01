# -*- coding: utf-8 -*-
"""
Parses an inferred policy obtained from rewards extracted by MaxEnt IRL
"""
import numpy as np
from collections import namedtuple

LinearVel = namedtuple('LinearVel',
["stop", "surge", "turn_ccw", "turn_cw"])
AngularVel = namedtuple('AngularVel',
["stop", "surge", "turn_ccw", "turn_cw"])

class SilkmothIRL:
    def __init__(self, fps, bin_edges, num_states, policy, rewards=None):

        self._tblank = 0.0
        self._hits_sum = 0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self._digi_tblank = 0
        self._digi_hits_sum = 0
        self.action = 0
        self.state = 0
        self.reward = 0
        self.bin_edges = bin_edges
        self.num_states = num_states
        self.policy = policy
        self.rewards = rewards

    def __str__(self):
        return 'tb:{:.3f}, H:{}, s:{}, a:{}'.format(
            self._tblank, self._hits_sum, self._digi_tblank, self.action)

    @property
    def log_header(self):
        return 'tblank,digi_tblank,hits,digi_hits,action'

    @property
    def log_step(self):
        return [self._tblank, self._digi_tblank, self._hits_sum, self._digi_hits_sum, self.action]

    # def get_state(self, ant_state):
    #     return self.digitize_tblank() + ant_state * (len(self.bin_edges) - 1)

    @property
    def action_space(self):
        lin_v = LinearVel(0.0, 19.0, 0.8, 0.8)
        ang_v = AngularVel(0.0, 0.062, 1.3, -1.3)

        u = {
            0: (lin_v.stop, ang_v.stop),
            1: (lin_v.surge, ang_v.surge),
            2: (lin_v.turn_ccw, ang_v.turn_ccw),
            3: (lin_v.turn_cw, ang_v.turn_cw)
        }

        return u

    def get_reward(self, digi_tblank, ant_state):
        # digi_tblank = self.digitize_tblank()
        return self.rewards[digi_tblank, ant_state]

    def hits_sum(self, ant_state):

        if ant_state > 0:
            self._hits_sum += 1

        return self._hits_sum

    def digitize_tblank(self):

        # Encode [antennae state, blank duration]
        # into an int according to the MaxEnt obtained policy
        # edges = self.bin_edges[self.num_states[0]].to_numpy()
        edges = self.bin_edges[self.num_states[0]].values
        # edges = self.bin_edges['tblank'].to_numpy()
        bins = len(edges) - 1
        # bins = len(self.bin_edges) - 1
        rtol = 1.e-5
        atol = 1.e-8
        eps = atol + rtol * self._tblank
        # d = np.digitize(self._tblank + eps, self.bin_edges[1:])
        d = np.digitize(self._tblank + eps, edges[1:])
        d = np.clip(d, 0, bins - 1)

        self._digi_tblank = d

        return self._digi_tblank, bins

    def digitize_hits_sum(self):

        # Encode [antennae state, blank duration]
        # into an int according to the MaxEnt obtained policy
        edges = self.bin_edges[self.num_states[1]].dropna().to_numpy()
        # print('Digi edges for hits: \n{}'.format(edges))
        bins = len(edges) - 1
        # bins = len(self.bin_edges) - 1
        rtol = 1.e-5
        atol = 1.e-8
        eps = atol + rtol * self._hits_sum
        # d = np.digitize(self._hits_sum + eps, self.bin_edges[1:])
        d = np.digitize(self._hits_sum + eps, edges[1:])
        d = np.clip(d, 0, bins - 1)

        self._digi_hits_sum = d

        return self._digi_hits_sum

    def random_action(self, ant_state, a):

        digi_tblank, bins = self.digitize_tblank()
        self.state = digi_tblank + ant_state * bins
        self.reward = self.get_reward(digi_tblank, ant_state)
        u = self.action_space
        self.linear_vel, self.angular_vel = u[a]

    def control(self, ant_state, last_hit):

        if ant_state > 0:
            self._tblank = .0
            self._hits_sum += 1

        digi_tblank, bins = self.digitize_tblank()
        self.action = np.argmax(self.policy[:, digi_tblank, ant_state])
        self.state = digi_tblank + ant_state * bins
        u = self.action_space
        self.linear_vel, self.angular_vel = u[self.action]