import numpy as np
from src.utils.geometry import Point
import src.controllers.infotaxis_core as core
from collections import namedtuple, deque


class Infotaxis(object):

    def __init__(self, fps, xlim, ylim, conf):

        self.dt = 1 / fps
        self.V = conf.V
        self.D = conf.D
        self.E = conf.E
        self.tau = conf.tau
        self.agent_size = conf.agent_size
        self.src_radius = conf.src_radius
        self._tblank = 0.0
        self.Ncells_x = conf.grid_shape[0]
        self.Ncells_y = conf.grid_shape[1]
        self.xbs = tuple([i / 1000 for i in xlim])
        # self.ybs = tuple([i / 1000 for i in ylim])
        self.ybs = (0, .720)
        self.xs = np.linspace(*self.xbs, self.Ncells_x)
        self.ys = np.linspace(*self.ybs, self.Ncells_y)
        # self.core = core(V, D, E, tau, agent_size)
        self.log_p_src = core.build_log_src_prior('uniform', self.xs, self.ys)
        self.agent_speed = conf.agent_speed
        self.entropy = core.entropy(self.log_p_src)
        self.delta_s_expected = 0
        self.entropies = deque(maxlen=10)
        self.hits = deque(maxlen=2 * fps)
        self.burstiness = 0
        self.hit_transitions = deque(maxlen=2)
        self.tb_on_hits = deque(maxlen=10)
        self.tb_B = 0
        self.dS_B = 0
        self.hit_B = 1
        self.wsum = 0
        self.dS_mean = 0
    def __str__(self):
        return 'wSum:{}, S:{:.4f}, tb_B:{:.4f}'.format(self.wsum, self.entropy,
                                                       self.tb_B)

    @property
    def log_header(self):
        return 'tblank,entropy,EDS,DSmean,wSum,hit_B,tb_B,ds_B'

    @property
    def log_step(self):
        return [
            self._tblank, self.entropy, self.delta_s_expected, self.dS_mean, self.wsum, self.hit_B, self.tb_B, self.dS_B
        ]

    def get_burstiness(self, x):
        B = (np.std(x) - np.mean(x)) / (np.std(x) + np.mean(x))
        return B

    def control(self, h, pos: Point):

        h = int(h > 0)

        pos = (pos.x / 1000, ((360 + pos.y) / 1000))

        self.log_p_src = core.update_log_p_src(
            pos, self.xs, self.ys, self.dt, h,
            self.V, self.D, self.E, self.agent_size, self.tau,
            self.src_radius, self.log_p_src)

        self.hits.append(h)
        self.hit_transitions.append(h)
        self.entropy = core.entropy(self.log_p_src)
        self.entropies.append(self.entropy)
        if len(self.entropies) >= 30:
            self.dS_mean = np.mean(np.diff(self.entropies))
            self.dS_B = self.get_burstiness(np.diff(self.entropies))

        if (len(self.hits) >= int(2 / (self.dt))) and (np.sum(self.hits) > 0):
            self.hit_B = self.get_burstiness(self.hits)

        if len(self.hit_transitions) > 1 and np.diff(
                self.hit_transitions)[0] > 0:
            self.tb_on_hits.append(self._tblank)
            self.wsum += 1

        if len(self.tb_on_hits) >= 10:
            self.tb_B = self.get_burstiness(self.tb_on_hits)

        if h:
            self._tblank = .0


        moves = core.get_moves(pos, self.xs, self.ys, (self.dt * self.agent_speed))
        delta_s_expecteds = []

        # get entropy decrease given src found
        delta_s_src_found = -self.entropy

        for move in moves:

            # set entropy increase to inf if move is out of bounds
            if not round(self.xbs[0], 6) <= round(move[0], 6) <= round(self.xbs[1], 6):
                delta_s_expecteds.append(np.inf)
                continue
            elif not round(self.ybs[0], 6) <= round(move[1], 6) <= round(self.ybs[1], 6):
                delta_s_expecteds.append(np.inf)
                continue

            # get probability of finding source
            p_src_found = core.get_p_src_found(move, self.xs, self.ys, self.log_p_src, self.src_radius)
            p_src_not_found = 1 - p_src_found

            # loop over probability and expected entropy decrease for each sample
            p_samples = np.nan * np.zeros(len([0, 1]))
            delta_s_given_samples = np.nan * np.zeros(len([0, 1]))

            for ctr, h in enumerate([0, 1]):

                # probability of sampling h at pos
                p_sample = core.get_p_sample(
                    pos=move, xs=self.xs, ys=self.ys, dt=self.dt, h=h,
                    w=self.V, d=self.D, r=self.E, a=self.agent_size, tau=self.tau, log_p_src=self.log_p_src)

                # posterior distribution from sampling h at pos
                log_p_src_ = core.update_log_p_src(
                    pos=move, xs=self.xs, ys=self.ys, dt=self.dt, src_radius=self.src_radius,
                    h=h, w=self.V, d=self.D, r=self.E, a=self.agent_size, tau=self.tau, log_p_src=self.log_p_src)

                # decrease in entropy for this move/sample
                s_ = core.entropy(log_p_src_)
                delta_s_given_sample = s_ - self.entropy

                p_samples[ctr] = p_sample
                delta_s_given_samples[ctr] = delta_s_given_sample

            # get expected entropy decrease given source not found
            delta_s_src_not_found = p_samples.dot(delta_s_given_samples)

            # compute total expected entropy decrease
            delta_s_expected = (p_src_found * delta_s_src_found) + \
                (p_src_not_found * delta_s_src_not_found)

            delta_s_expecteds.append(delta_s_expected)
            self.delta_s_expected = delta_s_expected


        try:
            best_action = moves[np.argmin(delta_s_expecteds)]
        except:
            best_action = moves[-1]
        return best_action
