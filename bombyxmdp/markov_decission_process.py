import numpy as np
import pandas as pd
import bombyxmdp.preprocessing as prep
import matplotlib.pyplot as plt

def print_full_dataframe(data):
    pd.set_option('display.max_rows', len(data))
    print(data)
    pd.reset_option('display.max_rows')


class NumericState(object):
    def __init__(self, name, bins, skewed, logscale, use_kmeans):
        self.name = name
        self.bins = bins
        self.skewed = skewed
        self.logscale = logscale
        self.use_kmeans = use_kmeans


class MothMDP(object):
    def __init__(self, df, numeric_states, categoric_states):
        self.df = df 
        self.numeric_states = numeric_states
        self.categoric_states = categoric_states
        self.num_state_bits = 0
        self.digi_edges = {}

    def digitize(self, X, edges):
        bins = len(edges) - 1
        rtol = 1.e-5
        atol = 1.e-8
        eps = atol + rtol * X
        X_d = np.digitize(X + eps, edges[1:])
        X_d = np.clip(X_d, 0, bins - 1)
        return X_d

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
        """
            The last bin of the digitalize state is define as
            from the quantile(0.98) to the max value
        """
        vmax = self.df[state.name].max()
        tail = self.df[state.name] > self.df[state.name].quantile(_quantile)
        vrange = self.df[~tail]
        bins = state.bins - 1

        states_k = [state.name, state.name + '_k']
        states_digi = [sk + '_digi' for sk in states_k]

        _, edges = prep.discretize(vrange[states_k], bins, strat_kmeans=state.use_kmeans)
        edges = np.append(edges[0], vmax)

        for i, state_digi in enumerate(states_digi):
            self.df.loc[:, state_digi] = self.digitize(self.df[states_k[i]].to_numpy(), edges)

        if state.logscale:
            edges = np.expm1(edges)
        return edges

    def digitize_numeric_states(self, state):
        if state.skewed:
            edges = self.digitize_skewed_state(state)
        else:
            edges = self.digitize_normal_state(state)

        if state.logscale:
            self.digi_edges[state.name.replace('log_', '')] = edges
        else:
            self.digi_edges[state.name] = edges

    # def merge_categoric_states(self):
    #     s0, s1 = tuple(self.categoric_states)
    #     sk0, sk1 = tuple([s + '_k' for s in self.categoric_states])
    #     b0, b1 = (len(self.df[s0].unique()), len(self.df[s1].unique()))
    #     print("hahahahaha", b0, b1)
    #     self.df.loc[:, 'state_cat_i'] = self.df[s0] + self.df[s1] * b0
    #     self.df.loc[:, 'state_cat_k'] = self.df[sk0] + self.df[sk1] * b0


    def merge_states(self):
        num_bits = self.num_state_bits
        self.df.loc[:, 'state_i'] = self.df['state_num_i'] + self.df['state_cat_i'] * num_bits
        self.df.loc[:, 'state_k'] = self.df['state_num_k'] + self.df['state_cat_k'] * num_bits

    def encode_states(self):
        state_obj = NumericState(*self.numeric_states)
        self.digitize_numeric_states(state_obj)
        state_name = state_obj.name
        ni, nk = (f'{state_name}_digi', f'{state_name}_k_digi')
        self.df.loc[:, 'state_num_i'] = self.df[ni]
        self.df.loc[:, 'state_num_k'] = self.df[nk]
        self.num_state_bits = state_obj.bins

        if len(self.categoric_states) > 1:
            # Merge categoric states
            self.merge_categoric_states()
        else:
            state = self.categoric_states[0]
            ci, ck = (state, f'{state}_k')
            self.df.loc[:, 'state_cat_i'] = self.df[ci]
            self.df.loc[:, 'state_cat_k'] = self.df[ck]

        # Merge numeric and categoric states
        self.merge_states()

    def encode_actions(self):
        """
            From linear velocity and angular velocity, encode the actions
            in each step time in to one of (surge, turn ccw, turn cw, stop)
        """
        lv_min = self.df['linear_vel'].mean() - self.df['linear_vel'].std()
        av_lo = self.df['angular_vel'].quantile(0.45)
        av_hi = self.df['angular_vel'].quantile(0.55)

        lv_min = self.df['linear_vel'].quantile(0.3)
        av_lo = -0.15
        av_hi = 0.15
        
        print(f'Min. linear vel. : {lv_min:.5f}')
        print(f'Angular vel. range: ({av_lo:.5f}, {av_hi:.5f})')

        surge = self.df['linear_vel'].gt(lv_min) \
            & self.df['angular_vel'].between(av_lo, av_hi, inclusive="both")

        # surge = self.df['linear_vel'].gt(lv_min) \
        #     & self.df['angular_vel'].between(-0.2, 0.2, inclusive="both")

        # surge = self.df['linear_vel'].gt(lv_min)

        turn_ccw = ~(surge) & (self.df['angular_vel'] > av_hi)
        turn_cw = ~(surge) & (self.df['angular_vel'] < av_lo)

        # stop = 0; surge = 1; turn ccw = 2, turn cw = 3
        for i, a in enumerate([surge, turn_ccw, turn_cw]):
            self.df.loc[a, 'action'] = i + 1
        self.df['action'].fillna(0, inplace=True)
        self.df['action'] = self.df.action.astype('uint8')

    def get_transition_probability(self):
        """
        Return the transition probability
        """
        t_ik = self.df.groupby(['state_i', 'action','state_k'])
        t_ik = t_ik.state_k.count()
        # print_full_dataframe(t_ik)
        n_states = len(self.df['state_cat_i'].unique()) * self.num_state_bits
        n_actions = len(t_ik.index.levels[1])

        # Create an array of levels for all possible transitions
        lvl = [range(n_states), range(n_actions), range(n_states)]
        new_index = pd.MultiIndex.from_product(lvl, names=t_ik.index.names)

        # Reindex the count and fill empty values with zero (NaN by default)
        t_ik = t_ik.reindex(new_index, fill_value=0)

        # Create a transition probability matrix
        t_ik = np.array(t_ik, dtype=np.float64)
        tp = t_ik.reshape(n_states, n_actions, n_states)

        # Normalize transition probability matrix for each (s,a)
        for i in range(tp.shape[0]):
            for j in range(tp.shape[1]):
                denom = np.sum(tp[i, j])
                tp[i, j] = tp[i, j] / denom if denom else 0

        # M = tp
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x,y,z = np.meshgrid(range(n_states), range(n_actions), range(n_states))
        # ax.scatter(x,y,z, c=M.flat)
        # plt.show()

        return tp