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

    def digitize_skewed_state(self, state, _quantile=.75):

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
        if state.logscale: edges = np.expm1(edges)

        for i, sd in enumerate(states_digi):
            self.df.loc[:, sd] = self._digitize(
                self.df[states_k[i]].to_numpy(), edges)

        return edges

    def digitize_numeric_states(self, states):

        for s in states:
            edges = self.digitize_numeric_state(s)
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

    def encode_actions_MG(self):
        tb, lin_vel, ang_vel = tuple(self.action_cols)

        av_ccw, av_cw = (0.087, -0.087)

        surge = (self.df[tb].le(0.5) & self.df[lin_vel].gt(0)) | (self.df[tb].gt(0.2) & self.df[ang_vel].between(av_cw, av_ccw, inclusive=False))

        turn_ccw = ~(surge) & (self.df[ang_vel] > 0)
        turn_cw = ~(surge) & (self.df[ang_vel] < 0)
        # stop = ~(surge | turn_ccw | turn_cw)

        # self.df.loc[:, 'action_mg'] = 0
        for i, a in enumerate([surge, turn_cw, turn_ccw]):
            self.df.loc[a, 'action_mg'] = i + 1

        self.df['action_mg'].fillna(0, inplace=True)
        self.df['action_mg'] = self.df.action_mg.astype('uint8')

    def encode_actions_KZ(self):
        tb, _, ang_vel = tuple(self.action_cols)

        # av_ccw, av_cw = (0.087, -0.087)

        surge = self.df[tb].le(0.5)
        # surge = self.df[tb].le(0.5) | self.df[ang_vel].eq(0)

        # turn_ccw = self.df[tb].gt(0.5) & (self.df[ang_vel] > 0)
        # turn_cw = self.df[tb].gt(0.5) & (self.df[ang_vel] < 0)
        # rotate = self.df[tb].gt(0.5)
        # rotate = ~surge
        turn_ccw = ~(surge) & (self.df[ang_vel] >= 0)
        turn_cw = ~(surge) & (self.df[ang_vel] < 0)
        # stop = self.df[ang_vel].eq(0) & self.df[lin_vel].eq(0)

        # self.df.loc[:, 'action_kz'] = 0
        for i, a in enumerate([surge, turn_cw, turn_ccw]):
            self.df.loc[a, 'action_kz'] = i + 1
            # self.df.loc[a, 'action_kz'] = i

        self.df['action_kz'].fillna(0, inplace=True)
        self.df['action_kz'] = self.df.action_kz.astype('uint8')

    def get_mismatching_expected_reward(self):

        self.df['match_pb'] = self.df.action_mg.eq(
            self.df.action_kz).astype('int')

        self.df['last_action_mg'] = self.df.loc[:, 'action_mg'].shift(1,
                                                                  fill_value=0)
        self.df['last_action_kz'] = self.df.loc[:, 'action_kz'].shift(1,
                                                                 fill_value=0)

        # self.df[:, 'expected_reward'] =

        rewards = ['r_stop', 'r_surge', 'r_turn_cw', 'r_turn_ccw']

        for i in sorted(self.df.action_kz.unique()):
            self.df.loc[(self.df.last_action_kz == i)
                        & self.df.match_pb, 'EX_DS'] = self.df[rewards[i]]


        for j in sorted(self.df.action_mg.unique()):
            self.df.loc[(self.df.last_action_mg == j)
                        & ~(self.df.match_pb), 'EX_DS'] = self.df[rewards[j]]

        # mat_sum = self.df[self.df.match_pb.eq(1)].EX_DS.sum()
        # mis_sum = self.df[self.df.match_pb.ne(1)].EX_DS.sum()

        # self.df.loc[self.df.match_pb.eq(1), 'EX_DS'] = self.df.EX_DS / mat_sum
        # self.df.loc[self.df.match_pb.ne(1), 'EX_DS'] = self.df.EX_DS / mis_sum

    # def get_mismatching_actual_reward(self):

    #     self.df['match_pb'] = self.df.action_mg.eq(
    #         self.df.action_kz).astype('int')

    #     self.df['last_action_mg'] = self.df.loc[:, 'action_mg'].shift(
    #         1, fill_value=0)
    #     self.df['last_action_kz'] = self.df.loc[:, 'action_kz'].shift(
    #         1, fill_value=0)

    #     # self.df[:, 'expected_reward'] =

    #     rewards = ['r_stop', 'r_surge', 'r_turn_cw', 'r_turn_ccw']

    #     for i in sorted(self.df.action_kz.unique()):
    #         self.df.loc[(self.df.last_action_kz == i)
    #                     & self.df.match_pb, 'DS'] = self.df[rewards[i]]

    #     for j in sorted(self.df.action_mg.unique()):
    #         self.df.loc[(self.df.last_action_mg == j)
    #                     & ~(self.df.match_pb), 'DS'] = self.df[rewards[j]]

    def get_planning_error(self):

        self.df.loc[:, 'epsilon'] = self.df.EX_DS.sub(self.df.DS)
        self.df['epsilon'] = self.df.epsilon**2

    def get_mismatching_probability(self):

        self.df.loc[:, 'pmat'] = self.df[self.df.match_pb.eq(1)].EX_DS
        self.df.loc[:, 'pmis'] = self.df[self.df.match_pb.ne(1)].EX_DS

        self.df['pmat'] = self.df.pmat / self.df.pmat.sum()
        self.df['pmis'] = self.df.pmis / self.df.pmis.sum()


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

    def get_ds_heatmap(self):

        t_ik = self.df.groupby(['hitsum', 'match_pb',
                                'DS']).DS

        n_states = len(self.df['hitsum'].unique())
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

        # tp = np.zeros((n_states, n_actions, n_states))
        # tp[:u_states, :, :u_states] = t_ik

        # # Normalize transition probability matrix for each action
        # for i in range(tp.shape[0]):
        #     for j in range(tp.shape[1]):

        #         tp[i, j] /= np.sum(tp[i, j])

        # tp = np.nan_to_num(tp)
        # print('Sum of tp[0]: {}'.format(np.sum(tp[0])))

        # self.u_states = u_states
        # self.u_actions = u_actions
        # self.n_states = n_states
        # self.n_actions = n_actions
        return tp

    @property
    def info(self):
        return 'Numeric states: {}\nCategoric states: {}\nBin edges: {}\n(u/n) states: {:.5f} ({}/{})\n(u/n) actions: ({}/{})'.format(
            self.numeric_states, self.categoric_states, self.digi_edges,
            (self.u_states / self.n_states), self.u_states, self.n_states,
            self.u_actions, self.n_actions)
