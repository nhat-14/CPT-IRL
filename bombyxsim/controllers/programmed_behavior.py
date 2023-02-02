"""
Silkmoth programmed behavior as defined by Kanzaki et al. (1992)
"""
from collections import namedtuple
import numpy as np

Delay = namedtuple('Delay', ['surge', 'turn1', 'turn2', 'turn3'])
LinearVel = namedtuple('LinearVel', ['stop', 'surge', 'turn'])
AngularVel = namedtuple('AngularVel', ['stop', 'surge', 'turnccw', 'turncw'])
AntStates = {'none': 0, 'right': 1, 'left': 2, 'both': 3}

class KPB(object):

    def __init__(self, fps):

        self.fps = fps
        self._tblank = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self._state = 'stop'

    def __str__(self):
        return 'tb:{:.3f}, s:{}'.format(self._tblank, self._state)

    @property
    def log_header(self):
        return 'tblank,state'

    @property
    def log_step(self):
        return [self._tblank, self._state]

    def tblank(self, ant_state):
        if ant_state <= 0:
            self._tblank += 1

        else: self._tblank = 0.0

        return self._tblank / self.fps

    def state(self, ant_state):

        delay = Delay(0.5, 1.2, 1.9, 2.1)

        if ant_state > 0:
            self._state = 'surge'
            self._tblank = .0

        else:
            if self._state == 'surge' and self._tblank >= delay.surge:
                self._state = 'turn1'
            if self._state == 'turn1' and self._tblank >= delay.turn1:
                self._state = 'turn2'
            if self._state == 'turn2' and self._tblank >= delay.turn2:
                self._state = 'turn3'
            if self._state == 'turn3' and self._tblank >= delay.turn3:
                self._state = 'loop'

        return self._state


    def control(self, ant_state, last_hit):
        state = self.state(ant_state)
        # lin_v = LinearVel(0.0, 26.0)
        # ang_v = AngularVel(0.0, 0.087, 1.0, -1.0) #radians per second
        lin_v = LinearVel(0.0, 19.0, 0.8)
        ang_v = AngularVel(0.0, 0.062, 1.3, -1.3)  #radians per second

        u = {
            'stop': lambda last_hit: (lin_v.stop, ang_v.stop),
            'surge': lambda last_hit: (
                lin_v.surge,
                ang_v.surge*-1 if last_hit == 0b01 else ang_v.surge),
            'turn1': lambda last_hit: (
                lin_v.turn,
                ang_v.turncw if last_hit == 0b10 else ang_v.turnccw
            ),
            'turn2': lambda last_hit: (
                lin_v.turn,
                ang_v.turncw if last_hit == 0b01 else ang_v.turnccw
            ),
            'turn3': lambda last_hit: (
                lin_v.turn,
                ang_v.turncw if last_hit == 0b10 else ang_v.turnccw
            ),
            'loop': lambda last_hit: (
                lin_v.turn,
                ang_v.turncw if last_hit == 0b01 else ang_v.turnccw
            )
        }

        self.linear_vel, self.angular_vel = u[state](last_hit)

    def control_Cesar(self, ant_state):
        action_list = {"stop":[0, 0],"surge":[19, 0.062],"turnccw":[0.8, 1.3],"turncw":[0.8, -1.3]}
        policy = [["surge","stop","stop","surge","turnccw","surge","turncw","turncw","turncw","stop","surge","turncw","turncw","stop","turnccw","turnccw","surge"],
        ["surge","turncw","turnccw","surge","turnccw","surge","turncw","turnccw","surge","turnccw","turncw","surge","stop","stop","stop","turncw","surge"],
        ["stop","turnccw","turnccw","turncw","stop","turnccw","surge","turncw","surge","stop","turnccw","turnccw","surge","stop","stop","turnccw","stop"],
        ["turnccw","turncw","turncw","turncw","turnccw","stop","turncw","turnccw","stop","turnccw","turnccw","turnccw","surge","stop","surge","stop","turnccw"]]
        blank_bins = [0.115, 0.371, 0.678, 1.012, 1.383, 1.823, 2.336, 2.914, 3.601, 4.446, 5.454, 6.672, 8.191, 10.104, 12.5, 32.533]
        dizi_blank = np.digitize(self._tblank,blank_bins)
        action = policy[ant_state][dizi_blank]
        self.linear_vel = action_list[action][0]
        self.angular_vel = action_list[action][1]

