import numpy as np
import pandas as pd
from collections import namedtuple

import preprocessing as prep

class NumericState(object):
    def __init__(self, name, bins, skewed, logscale, use_kmeans):

        self.name = name
        self.bins = bins
        self.skewed = skewed
        self.logscale = logscale
        self.use_kmeans = use_kmeans


class MothMDP(object):
    def __init__(self, df, numeric_states, categoric_states, action_cols):

        self.df = df
        self.numeric_states = numeric_states
        self.categoric_states = categoric_states
        self.action_cols = action_cols
        self.num_state_bits = 0
        self.digi_edges = {}
        self.n_states = 0
        self.n_actions = 0
        self.u_states = 0
        self.u_actions = 0

    def _digitize(self, X, edges):

        bins = len(edges) - 1
        rtol = 1.e-5
        atol = 1.e-8
        eps = atol + rtol * X

        X_d = np.digitize(X + eps, edges[1:])
        X_d = np.clip(X_d, 0, bins - 1)

        return X_d

    def digitize_numeric_state(self, state):

        if state.skewed:
            return self.digitize_skewed_state(state)

        else:
            return self.digitize_normal_state(state)

    def digitize_normal_state(self, state):

        states_k = [state.name, state.name + '_k']
        states_digi = [sk + '_digi' for sk in states_k]

        x, edges = prep.discretize(self.df[states_k],
                                   state.bins,
                                   strat_kmeans=state.use_kmeans)

        edges = edges[0]
        for i, sd in enumerate(states_digi):
            self.df.loc[:, sd] = x[:, i].astype('int')

        return edges

    def digitize_skewed_state(self, state, _quantile=.98):

        vmax = self.df[state.name].max()
        tail = self.df[state.name] > self.df[state.name].quantile(_quantile)
        vrange = self.df[~tail]
        bins = state.bins - 1

        states_k = [state.name, state.name + '_k']
        states_digi = [sk + '_digi' for sk in states_k]

        _, edges = prep.discretize(vrange[states_k],
                                   bins,
                                   strat_kmeans=state.use_kmeans)
        edges = np.append(edges[0], vmax)

        for i, sd in enumerate(states_digi):
            self.df.loc[:, sd] = self._digitize(
                self.df[states_k[i]].to_numpy(), edges)

        if state.logscale: edges = np.expm1(edges)
        return edges

    def digitize_numeric_states(self, states):

        for s in states:
            edges = self.digitize_numeric_state(s)
            if s.logscale:
                self.digi_edges[s.name.replace('log_', '')] = edges
            else:
                self.digi_edges[s.name] = edges

    def merge_numeric_states(self, states):

        s0, s1 = tuple([s.name + '_digi' for s in states])
        sk0, sk1 = tuple([s.name + '_k_digi' for s in states])
        b0, b1 = (states[0].bins, states[1].bins)

        self.df.loc[:, 'state_num_i'] = self.df[s0] + self.df[s1] * b0
        self.df.loc[:, 'state_num_k'] = self.df[sk0] + self.df[sk1] * b0

        self.num_state_bits = b0 * b1

    def merge_categoric_states(self):

        s0, s1 = tuple(self.categoric_states)
        sk0, sk1 = tuple([s + '_k' for s in self.categoric_states])
        b0, b1 = (len(self.df[s0].unique()), len(self.df[s0].unique()))

        self.df.loc[:, 'state_cat_i'] = self.df[s0] + self.df[s1] * b0
        self.df.loc[:, 'state_cat_k'] = self.df[sk0] + self.df[sk1] * b0

    def merge_states(self):

        ni, nk = ('state_num_i', 'state_num_k')
        ci, ck = ('state_cat_i', 'state_cat_k')
        num_bits = self.num_state_bits

        self.df.loc[:, 'state_i'] = self.df[ni] + self.df[ci] * num_bits
        self.df.loc[:, 'state_k'] = self.df[nk] + self.df[ck] * num_bits

    def encode_states(self):

        _numeric_states = []
        for key in self.numeric_states:
            _numeric_states.append(NumericState(*self.numeric_states[key]))

        # Digitize numeric states
        self.digitize_numeric_states(_numeric_states)

        if len(self.numeric_states) > 1:
            # Merge numeric states
            self.merge_numeric_states(_numeric_states)

        else:
            print(f'numeric state: {self.numeric_states[0][0]}')
            ni, nk = (self.numeric_states[0][0] + '_digi',
                      self.numeric_states[0][0] + '_k_digi')
            self.df.loc[:, 'state_num_i'] = self.df[ni]
            self.df.loc[:, 'state_num_k'] = self.df[nk]
            self.num_state_bits = self.numeric_states[0][1]

        if len(self.categoric_states) > 1:
            # Merge categoric states
            self.merge_categoric_states()

        else:
            ci, ck = (self.categoric_states[0],
                      self.categoric_states[0] + '_k')
            self.df.loc[:, 'state_cat_i'] = self.df[ci]
            self.df.loc[:, 'state_cat_k'] = self.df[ck]

        # Merge numeric and categoric states
        self.merge_states()

    def encode_actions(self, verbose=False):
        lin_vel, ang_vel = tuple(self.action_cols)
        lv_min = self.df[lin_vel].mean() - self.df[lin_vel].std()
        # lv_lo, lv_hi = (self.df[lin_vel].quantile(0.45),
        # self.df[lin_vel].quantile(0.55))
        # av_lo, av_hi = (-self.df[ang_vel].mean(), self.df[ang_vel].mean())
        # av_lo, av_hi = (-.087, .087)
        av_lo, av_hi = (self.df[ang_vel].quantile(0.40),
                        self.df[ang_vel].quantile(0.60))

        if verbose:

            print('Min. linear vel. : {:.5f}'.format(lv_min))
            # print('Linear vel. range: ({:.5f}, {:.5f})'.format(lv_lo, lv_hi))
            print('Angular vel. range: ({:.5f}, {:.5f})'.format(av_lo, av_hi))

        # stop = (self.df[lin_vel] < lv_min) & (self.df[ang_vel].between(
        #     av_lo, av_hi, inclusive=True))
        surge = self.df[lin_vel].gt(lv_min) & self.df[ang_vel].between(
            av_lo, av_hi, inclusive=True)
        turn_ccw = ~(surge) & (self.df[ang_vel] > av_hi)  # & (self.df[lin_vel] >= lv_min)
        turn_cw = ~(surge) & (self.df[ang_vel] < av_lo)  # & (self.df[lin_vel] >= lv_min)

        for i, a in enumerate([surge, turn_ccw, turn_cw]):
            self.df.loc[a, 'action'] = i + 1

        # self.df.loc[stop, 'action'] = 0
        self.df['action'].fillna(0, inplace=True)
        self.df['action'] = self.df.action.astype('uint8')

    def encode_many_actions(self, verbose=False):
        lin_vel, ang_vel = tuple(self.action_cols)
        lv_min = self.df[lin_vel].mean() - self.df[lin_vel].std()
        # lv_lo, lv_hi = (self.df[lin_vel].quantile(0.45),
        # self.df[lin_vel].quantile(0.55))
        # av_lo, av_hi = (-self.df[ang_vel].mean(), self.df[ang_vel].mean())
        # av_lo, av_hi = (-.087, .087)
        av_lo, av_l, av_h, av_hi = (self.df[ang_vel].quantile(0.20),
                                    self.df[ang_vel].quantile(0.40),
                                    self.df[ang_vel].quantile(0.60),
                                    self.df[ang_vel].quantile(0.80))

        if verbose:

            print('Min. linear vel. : {:.5f}'.format(lv_min))
            # print('Linear vel. range: ({:.5f}, {:.5f})'.format(lv_lo, lv_hi))
            print('Angular vel. range: ({:.5f}, {:.5f})'.format(av_lo, av_hi))

        # stop = (self.df[lin_vel] < lv_min) & (self.df[ang_vel].between(
        #     av_lo, av_hi, inclusive=True))
        surge = self.df[lin_vel].gt(lv_min) & self.df[ang_vel].between(
            av_lo, av_hi, inclusive=True)
        turn_ccw = ~(surge) & (self.df[ang_vel] > av_hi)
        turn_left = ~(surge) & (self.df[ang_vel].between(
            av_h, av_hi, inclusive=True))
        turn_right = ~(surge) & (self.df[ang_vel].between(
            av_lo, av_l, inclusive=True))
        turn_cw = ~(surge) & (self.df[ang_vel] < av_lo)

        for i, a in enumerate(
            [surge, turn_ccw, turn_left, turn_right, turn_cw]):
            self.df.loc[a, 'action'] = i + 1

        # self.df.loc[stop, 'action'] = 0
        self.df['action'].fillna(0, inplace=True)
        self.df['action'] = self.df.action.astype('uint8')

    def get_transition_probability(self):

        t_ik = self.df.groupby(['state_i', 'action',
                                'state_k']).state_k.count()

        n_states = len(self.df['state_cat_i'].unique()) * self.num_state_bits
        n_actions = len(t_ik.index.levels[1])

        _u_states = t_ik.index.levels[0].values
        _u_actions = t_ik.index.levels[1].values

        u_states = len(_u_states)
        u_actions = len(_u_actions)

        # Create an array of levels for all possible transitions
        lvl = [
            t_ik.index.levels[0].values, t_ik.index.levels[1].values,
            range(u_states)
        ]
        new_index = pd.MultiIndex.from_product(lvl, names=t_ik.index.names)

        # Reindex the count and fill empty values with zero (NaN by default)
        t_ik = t_ik.reindex(new_index, fill_value=0)

        # Create a transition probability matrix
        t_ik = np.array(t_ik,
                        dtype=np.float64).reshape(u_states, u_actions,
                                                  u_states)

        tp = np.zeros((n_states, n_actions, n_states))
        tp[:u_states, :, :u_states] = t_ik

        # Normalize transition probability matrix for each action
        for i in range(tp.shape[0]):
            for j in range(tp.shape[1]):

                tp[i, j] /= np.sum(tp[i, j])

        tp = np.nan_to_num(tp)
        print('Sum of tp[0]: {}'.format(np.sum(tp[0])))

        self.u_states = u_states
        self.u_actions = u_actions
        self.n_states = n_states
        self.n_actions = n_actions
        return tp

    @property
    def info(self):
        return 'Numeric states: {}\nCategoric states: {}\n(u/n) states: {:.5f} ({}/{})\n(u/n) actions: ({}/{})'.format(
            self.numeric_states, self.categoric_states,
            (self.u_states / self.n_states), self.u_states, self.n_states,
            self.u_actions, self.n_actions)
