import numpy as np
import math
from utils.geometry import Point
import controllers.infotaxis_core as core
from collections import namedtuple, deque

Delay = namedtuple('Delay', ['surge', 'turn1', 'turn2', 'turn3'])
LinearVel = namedtuple('LinearVel', ['stop', 'surge', 'turn'])
AngularVel = namedtuple('AngularVel', ['stop', 'surge', 'turnccw', 'turncw'])
AntStates = {'none': 0, 'right': 1, 'left': 2, 'both': 3}


class C2R(object):
    def __init__(self, fps, xlim, ylim, conf):

        self.dt = 1 / fps
        self.V = conf.V# core.V
        self.D = conf.D# core.D
        self.E = conf.E# core.E
        self.tau = conf.tau# core.TAU
        self.agent_size = conf.agent_size# core.AGT_SIZE
        self.src_radius = conf.src_radius# core.SRC_RADIUS
        self.Ncells_x = conf.grid_shape[0]# core.NCX
        self.Ncells_y = conf.grid_shape[1]# core.NCY
        self.xbs = tuple([i / 1000 for i in xlim])
        # self.ybs = tuple([i / 1000 for i in ylim])
        self.ybs = (0, .720)
        self.xs = np.linspace(*self.xbs, self.Ncells_x)
        self.ys = np.linspace(*self.ybs, self.Ncells_y)
        self.log_p_src = core.build_log_src_prior('uniform', self.xs, self.ys)
        self.agent_speed = conf.agent_speed# core.AGT_SPEED
        self.entropy = core.entropy(self.log_p_src)
        self.S0 = self.entropy
        self.prev_entropy = self.entropy
        self.cumsum_ds_on_hit = 0
        self._tblank = 0.0
        self._hits_sum = 0
        self.wsum = 0
        self.delta_s_expected = 0
        self.qlen = fps * 2
        self.entropy_on_hits = deque(maxlen=30)
        self.hit_transitions = deque(maxlen=2)
        self.tb_on_hits = deque(maxlen=10)
        self.hit_B = 1
        self.hitsQ = deque(maxlen=self.qlen)
        self.tb_std = 100
        self.tb_B = 0
        self.entropy_rms = 100
        self.policy_threshold = 1e-3
        self.policy_mode = 'ITX'
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self._kpb_phase = 'stop'

    def __str__(self):
        return 'S:{:.4f}, tb:{:.2f}, hsum:{:d}, hB:{:.4f}, tbB:{:.4f}, M:{}'.format(
            self.entropy, self._tblank, self.wsum, self.hit_B, self.tb_B,
            self.policy_mode)

    @property
    def log_header(self):
        return 'tblank,entropy,ExDS,S_rms,wSum,hit_B,tb_B,mode'

    @property
    def log_step(self):
        return [
            self._tblank, self.entropy, self.delta_s_expected,
            self.entropy_rms, self.wsum, self.hit_B, self.tb_B,
            self.policy_mode
        ]

    def get_policy_mode(self):

        p = self._sigmoid(self.wsum * (self.tb_B + .2))
        if p > .5:
            self.policy_mode = 'KPB'

        else:
            self.policy_mode = 'ITX'

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def kpb_phase(self, ant_state):

        delay = Delay(0.5, 1.2, 1.9, 2.1)

        if ant_state > 0:
            self._kpb_phase = 'surge'
            self._tblank = .0

        else:
            if self._kpb_phase == 'surge' and self._tblank >= delay.surge:
                self._kpb_phase = 'turn1'
            if self._kpb_phase == 'turn1' and self._tblank >= delay.turn1:
                self._kpb_phase = 'turn2'
            if self._kpb_phase == 'turn2' and self._tblank >= delay.turn2:
                self._kpb_phase = 'turn3'
            if self._kpb_phase == 'turn3' and self._tblank >= delay.turn3:
                self._kpb_phase = 'loop'

        return self._kpb_phase

    # def get_memory_coeff(self, x):
    #     Nx = len(x)
    #     mu = np.mean(x)

    def get_burstiness(self, x):
        B = (np.std(x) - np.mean(x)) / (np.std(x) + np.mean(x))
        return B

    def control(self, h, last_hit, pos: Point):

        pos_itx = (pos.x / 1000, ((360 + pos.y) / 1000))
        self.hit_transitions.append(int(h > 0))
        self.hitsQ.append(int(h > 0))

        self.log_p_src = core.update_log_p_src(pos_itx, self.xs, self.ys,
                                               self.dt, h, self.V, self.D,
                                               self.E, self.agent_size,
                                               self.tau, self.src_radius,
                                               self.log_p_src)

        self.entropy = core.entropy(self.log_p_src)

        if (len(self.hitsQ) >= self.qlen) and (np.sum(self.hitsQ) > 0):
            self.hit_B = self.get_burstiness(self.hitsQ)

        self.entropy_on_hits.append(self.entropy)
        if len(self.entropy_on_hits) >= 30:
            # print(self.entropy_on_hits)
            entropy_sqdiff = np.diff(np.array(self.entropy_on_hits))**2
            # S_on_h = np.array(self.entropy_on_hits)[:-1] - np.array(
            # self.entropy_on_hits)[1:]
            # self.entropy_rms = np.sqrt(np.mean(S_on_h**2))
            self.entropy_rms = np.sqrt(np.mean(entropy_sqdiff))

        if len(self.hit_transitions) > 1 and np.diff(
                self.hit_transitions)[0] > 0:
            self.tb_on_hits.append(self._tblank)
            self.wsum += 1
        if len(self.tb_on_hits) >= 10:
            self.tb_B = self.get_burstiness(self.tb_on_hits)

        if h > 0:
            self._tblank = .0
            self._hits_sum += 1
            # Choose policy mode
            self.get_policy_mode()

        self.prev_entropy = self.entropy


        if self.policy_mode == 'ITX':
            return self.itx_control(h, last_hit, pos_itx)

        elif self.policy_mode == 'KPB':
            return self.kpb_control(h, last_hit, pos)

        else: return 0

    def itx_control(self, h, last_hit, pos: Point):

        h = int(h > 0)

        moves = core.get_moves(pos, self.xs, self.ys,
                               (self.dt * self.agent_speed))
        delta_s_expecteds = []

        # get entropy decrease given src found
        delta_s_src_found = -self.entropy

        for move in moves:

            # set entropy increase to inf if move is out of bounds
            if not round(self.xbs[0], 6) <= round(move[0], 6) <= round(
                    self.xbs[1], 6):
                delta_s_expecteds.append(np.inf)
                continue
            elif not round(self.ybs[0], 6) <= round(move[1], 6) <= round(
                    self.ybs[1], 6):
                delta_s_expecteds.append(np.inf)
                continue

            # get probability of finding source
            p_src_found = core.get_p_src_found(move, self.xs, self.ys,
                                               self.log_p_src, self.src_radius)
            p_src_not_found = 1 - p_src_found

            # loop over probability and expected entropy decrease for each sample
            p_samples = np.nan * np.zeros(len([0, 1]))
            delta_s_given_samples = np.nan * np.zeros(len([0, 1]))

            for ctr, h in enumerate([0, 1]):

                # probability of sampling h at pos
                p_sample = core.get_p_sample(pos=move,
                                             xs=self.xs,
                                             ys=self.ys,
                                             dt=self.dt,
                                             h=h,
                                             w=self.V,
                                             d=self.D,
                                             r=self.E,
                                             a=self.agent_size,
                                             tau=self.tau,
                                             log_p_src=self.log_p_src)

                # posterior distribution from sampling h at pos
                log_p_src_ = core.update_log_p_src(pos=move,
                                                   xs=self.xs,
                                                   ys=self.ys,
                                                   dt=self.dt,
                                                   src_radius=self.src_radius,
                                                   h=h,
                                                   w=self.V,
                                                   d=self.D,
                                                   r=self.E,
                                                   a=self.agent_size,
                                                   tau=self.tau,
                                                   log_p_src=self.log_p_src)

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

    def kpb_control(self, h, last_hit, pos: Point):

        kpb_phase = self.kpb_phase(h)
        # lin_v = LinearVel(0.0, 19.0, 0.8)
        lin_v = LinearVel(0.0, 20.0, 0.8)
        # lin_v = LinearVel(0.0, (core.AGT_SPEED * 1e3) / 2, 0.8)
        ang_v = AngularVel(0.0, 0.062, 1.3, -1.3)  #radians per second

        u = {
            'stop':
            lambda last_hit: (lin_v.stop, ang_v.stop),
            'surge':
            lambda last_hit: (lin_v.surge, ang_v.surge * -1
                              if last_hit == 0b01 else ang_v.surge),
            'turn1':
            lambda last_hit: (lin_v.turn, ang_v.turncw
                              if last_hit == 0b10 else ang_v.turnccw),
            'turn2':
            lambda last_hit: (lin_v.turn, ang_v.turncw
                              if last_hit == 0b01 else ang_v.turnccw),
            'turn3':
            lambda last_hit: (lin_v.turn, ang_v.turncw
                              if last_hit == 0b10 else ang_v.turnccw),
            'loop':
            lambda last_hit: (lin_v.turn, ang_v.turncw
                              if last_hit == 0b01 else ang_v.turnccw)
        }

        # self.linear_vel, self.angular_vel = u[kpb_phase](last_hit)
        return u[kpb_phase](last_hit)