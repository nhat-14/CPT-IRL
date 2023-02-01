#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze logs from bombyxsim
IN: path of directory containing log files (csv format)
"""

from tqdm import tqdm
# from scipy import stats
from os.path import isfile, join
from sklearn.preprocessing import PowerTransformer, RobustScaler, KBinsDiscretizer
from scipy import stats
from scipy.spatial.distance import cdist
from pathlib import Path
# import tunnel_bombyx.utils as utils
import argparse
import datetime
import sys
import h5py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
sns.set(font="sans-serif", rc={"font.sans-serif": ["FreeSans"]})
import os
import glob
import json
import mothVR.preprocessing as preproc
import mothVR.mdp as mdp
from utils import fileIO, mdp_plots

from mothVR import __version__

__author__ = "Cesar Hernandez"
__copyright__ = "Cesar Hernandez"
__license__ = "mit"

_logger = logging.getLogger(__name__)

np.random.seed(99)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description=
        "Generate state-action trajectories from Shigaki's 2020 W+O tethered system logs"
    )
    parser.add_argument('--version',
                        action='version',
                        version='tunnel_bombyx {ver}'.format(ver=__version__))
    parser.add_argument("-p",
                        "--plot",
                        dest="plot",
                        help='Plot reward and action-value function',
                        type=str,
                        nargs='+')
    # const='plot.png')
    parser.add_argument("-i",
                        "--input-dir",
                        type=str,
                        dest="input_dir",
                        help='Path of the directory with the input data',
                        required=True)
    parser.add_argument("-D",
                        "--demos",
                        type=str,
                        dest="demos",
                        help='Path of the directory with expert demonstrations',
                        required=False)
    parser.add_argument("--compare",
                        type=str,
                        dest="compare",
                        help='Name of folders to compare',
                        nargs='+',
                        required=False)
    parser.add_argument("-c",
                        "--controller",
                        type=str,
                        dest="controller",
                        help='Type of controller used by the agent',
                        default='IRL',
                        required=False)
    parser.add_argument("--save-excel",
                        dest="save_excel",
                        help='Save descriptive stats to xlsx',
                        nargs="?",
                        const='summary')
    parser.add_argument("--save-trans-prob",
                        dest="save_trans_prob",
                        help='Save transition probabilities to .npy',
                        nargs="?",
                        const='trans_prob')
    parser.add_argument("--save-csv",
                        dest="save_csv",
                        help='Save merged dataframe to csv',
                        nargs='?',
                        const='rldemos')
    parser.add_argument('--stat-test',
                        dest="stat_test",
                        help="Test statistical significance",
                        action='store_true',
                        default=0)
    parser.add_argument('-v',
                        '--verbose',
                        dest="loglevel",
                        help="set loglevel to INFO",
                        action='store_const',
                        const=logging.INFO)
    parser.add_argument('-vv',
                        '--very-verbose',
                        dest="loglevel",
                        help="set loglevel to DEBUG",
                        action='store_const',
                        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=loglevel,
                        stream=sys.stdout,
                        format=logformat,
                        datefmt="%Y-%m-%d %H:%M:%S")

def centerline_deviation(y, dt, y_src=0.):

    y = y.to_numpy()
    dy = (y - y_src) * dt

    c = np.cumsum(dy)

    return pd.Series(c)

def traveled_distance(x, y):

    x = x.to_numpy()
    y = y.to_numpy()

    dist = np.hypot((x[:-1] - x[1:]), (y[:-1] - y[1:]))
    dist = np.insert(dist, 0, 0)

    return pd.Series(dist)


def get_traveled_distance(df, xcol, ycol):

    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    xy = np.array([x, y]).transpose()

    d_r = 0
    for i in range(xy.T[0].size):
        d_r = d_r + np.linalg.norm(xy[i] - xy[i - 1])

    # d_r = d_r/1000.0		# Convert cm to m
    return d_r


def modified_hausdorff(A, B):

    D = cdist(A, B)
    fhd = np.mean(np.min(D, axis=0))
    rhd = np.mean(np.min(D, axis=1))
    mhd = max(fhd, rhd)
    return mhd

def get_mhd(demos, sims):

    mhds = []
    indexes = lambda df, N: np.random.randint(0, len(df), N)

    for d in tqdm(demos, leave=False):
        dfd = pd.read_csv(d, usecols=['state_i', 'action'])
        # dfd = pd.read_csv(d, usecols=['tblank', 'antennae'])
        # demo = dfd.to_numpy()
        demo = dfd.iloc[indexes(demo, 1000)].to_numpy()

        for i, s in sims.groupby((sims.Time.diff() < 0).cumsum()):
            sim = s[['state_i', 'action']].to_numpy()
            # sim = s[['state', 'action']].iloc[indexes(sim, 1000)].to_numpy()
            # sim = s[['tblank', 'antennae']].to_numpy()
            mhds.append(modified_hausdorff(sim, demo))
            # mhds.append(modified_hausdorff(sim, demo) / len(sim))

    mhds = pd.Series(mhds)
    return mhds.mean()


def get_revised_mhd(demos, sims):

    mhds = []
    # indexes = lambda df, N: np.random.randint(0, len(df), N)

    for j, d in tqdm(demos.groupby((demos.Time.diff() < 0).cumsum()),
                     leave=False):
        # dfd = pd.read_csv(d, usecols=['state_i', 'action'])
        # dfd = pd.read_csv(d, usecols=['tblank', 'antennae'])
        # demo = dfd.to_numpy()
        # demo = dfd.iloc[indexes(demo, 1000)].to_numpy()

        for i, s in sims.groupby((sims.Time.diff() < 0).cumsum()):
            # indexes = np.random.randint(0, min([len(d), len(s)]), 1000)
            # indexes = np.arange(0, min([len(d), len(s)]), 3)
            demo = d[['x', 'y']].to_numpy()
            sim = s[['x', 'y']].to_numpy()
            # demo = d[['state_i', 'action']].to_numpy()
            # sim = s[['state', 'action']].to_numpy()
            # sim = s[['state', 'action']].iloc[indexes(sim, 1000)].to_numpy()
            # sim = s[['tblank', 'antennae']].to_numpy()
            # mhds.append(modified_hausdorff(sim, demo))
            mhds.append(modified_hausdorff(sim, demo) / (len(s) / 30))

    mhds = pd.Series(mhds)
    return mhds
    # return mhds.mean()


def RMSPE(x, y):

    # if x.shape[0] != y.shape[0]:
    #     print(f'Shape of y: {y.shape[0]}')
    #     y = y[:(x.shape[0] - 1)]
    #     print(f'Shape of x: {x.shape[0]}')

    # l = sorted((x, y), key=len)
    # c = l[1].copy()
    # c[:len(l[0])] -= l[0]

    rms = np.sqrt(np.sum(((x - y))**2) / np.sum(x**2))
    # rms = np.sqrt(np.sum((c)**2) / np.sum(x**2))
    # print(f'X: {x.shape}; Y:{y.shape}; RMSE: {(c).shape}')

    return rms

def get_RMS_error(demos, sims, x):

    errors = []

    for d in tqdm(demos, leave=False):
        dfd = pd.read_csv(d, usecols=[x])

        for i, s in sims.groupby((sims.Time.diff() < 0).cumsum()):
            sim = s.loc[0:len(dfd), x]
            errors.append(RMSPE(dfd.to_numpy(), sim.to_numpy()))

    errors = pd.Series(errors)
    return errors.mean()

def NLL(x):

    # print(len(x))
    a = np.clip(x, 1e-3, 0.999)
    # print('[{}, {}]'.format(np.min(a), np.max(a)))
    nll = -np.log(np.product(a))
    nll = np.sum(nll) / len(a)

    return nll


def negative_log_loss(demos):

    NLLs = []

    for j, d in tqdm(demos.groupby((demos.Time.diff() < 0).cumsum()),
                     leave=False):

        # print(d['pi_s_a'].head(10))
        indexes = np.random.randint(0, len(d), 200)
        NLLs.append(NLL(d['pi_s_a'].iloc[indexes].to_numpy()))
        # for i, s in sims.groupby((sims.Time.diff() < 0).cumsum()):

    NLLs = pd.Series(NLLs)
    return NLLs.mean()


def get_reward_RMSPE(demos, bins, reward, sims):

    errors = []

    # demo_dfs, _ = merge_data(demos,
    #                          ctrl='MOTH',
    #                          bins=bins,
    #                          reward_func=reward,
    #                          ignore_fails=False)



    # indexes = lambda df, N: np.random.randint(0, len(df), N)
    for j, d in tqdm(demos.groupby((demos.Time.diff() < 0).cumsum()),
                     leave=False):
        # exd = d.loc[:, 'tblank']
        for i, s in sims.groupby((sims.Time.diff() < 0).cumsum()):
            # sim = s.loc[0:len(exd), 'tblank']
            # sim = s.loc[:, 'reward']
            # indexes = np.random.randint(0, min([len(d), len(s)]), 1000)
            indexes = np.arange(0, min([len(d), len(s)]))
            errors.append(
                RMSPE(d['tblank'].iloc[indexes].values,
                      s['tblank'].iloc[indexes].values) / (len(s) / 30))
            # errors.append(RMSPE(d.tblank, s.tblank))

    errors = pd.Series(errors)
    return errors
    # return errors.mean()


def reached_goal(x, y, radius):

    return (np.sqrt(x**2 + y**2) < radius)

def plot_entropyrms_map(df):

    df['antennae'] = (df['antennae'].gt(0)).astype('uint8')
    # df['S_rms'] = df.S_rms*10
    sns.set(style='ticks', context='paper')
    fig, ax = plt.subplots(figsize=(6, 7.2))
    ax = sns.scatterplot(data=df[df.antennae > 0],
                         x='x',
                         y='y',
                         size='S_rms',
                         hue='antennae')
    ax.set_xlim(0, 600)
    ax.set_ylim(-360, 360)
    plt.show()

def compare_experiments(experiments, ignore_fails=True, ignore_idx=True):

    ex1, ex2 = (experiments[1], experiments[3])
    ctrl1, ctrl2 = (experiments[0], experiments[2])

    df1, _, _, _ = merge_data(ex1, ignore_fails=ignore_fails, ctrl=ctrl1)
    df2, _, _, _ = merge_data(ex2, ignore_fails=ignore_fails, ctrl=ctrl2)
    df = pd.concat([df1.assign(Controller=ctrl1), df2.assign(Controller=ctrl2)])

    return df

    # sns.set(style='ticks', context='paper')
    # fig, ax = plt.subplots()

    # ax = sns.lineplot(x='Time', y='entropy', hue='ctrl', data=df)
    # ax = sns.lineplot(x='Time', y='entropy', data=df2)

    # plt.show()

def compare_experiments_stats(experiments, ignore_fails=True, ignore_idx=True):

    ex1, ex2 = (experiments[1], experiments[3])
    ctrl1, ctrl2 = (experiments[0], experiments[2])

    df1, sr1, st1, tt1 = merge_data(ex1, ignore_fails=ignore_fails, ctrl=ctrl1)
    df2, sr2, st2, tt2 = merge_data(ex2, ignore_fails=ignore_fails, ctrl=ctrl2)

    fisher, pval_sr = stats.fisher_exact(np.stack((sr1, sr2)).T)
    mwst, pv_st = stats.mannwhitneyu(st1, st2)
    mwtt, pv_tt = stats.mannwhitneyu(tt1, tt2)
    print(pval_sr)
    print(pv_st)
    print(pv_tt)
    # df = pd.concat(
    #     [df1.assign(Controller=ctrl1),
    #      df2.assign(Controller=ctrl2)])


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def merge_data(path_,
               ctrl,
               bins=None,
               reward_func=None,
               policy=None,
               ignore_idx=True,
               ignore_fails=True,
               timeout=0):
    # Get paths of csv files
    csvs = [i for i in glob.glob(os.path.join(path_, '*.csv'))]
    _logger.info('Processing {}'.format(path_))
    print('Controller: {}'.format(ctrl))
    # csvs = [csvs[0], csvs[-1]]

    # Store dataframes in a list
    dfs = []
    lengths = []
    success_rate = 0
    success_time = []
    tortuosities = []

    for c in tqdm(csvs, ncols=0, desc='Merging csvs'):
        # Read csv into a dataframe
        df = pd.read_csv(c)

        if ctrl == 'IRL':
            columns = [
                'Time', 'antennae', 'x', 'y', 'theta', 'tblank', 'digi_tblank', 'hits',
                'digi_hits', 'action'
            ]
        elif ctrl == 'KPB':
            columns = ['Time', 'antennae', 'x', 'y', 'theta', 'tblank', 'kpb_phase']

        elif ctrl == 'ITX':
            columns = [
                'Time', 'antennae', 'x', 'y', 'theta', 'tblank', 'entropy',
                'EDS', 'DS_mean', 'wSum', 'hit_B', 'tb_B', 'dS_B'
            ]

        elif ctrl == 'HYB':
            columns = [reward
                'Time', 'antennae', 'x', 'y', 'theta', 'tblank', 'entropy',
                'EDS', 'S_rms', 'wSum', 'hit_B', 'tb_B', 'mode'
            ]

        elif ctrl == 'MOTH':
            columns = ['Time', 'x', 'y', 'linear_vel', 'angular_vel','tblank', 'log_twhiff', 'lasthit', 'theta', 'wind', 'antennae', 'state_num_i', 'state_i', 'action']

        df.columns = columns

        # Define time step duration as a convenience variable
        tstep = df['Time'].iloc[1]

        # Unwrap theta to avoid abrupt changes from 2*pi to 0
        theta = df['theta'].to_numpy()
        df['theta'] = theta

        if bins is not None:
            df.loc[:, 'digi_tblank'] = digitize_tblank(df.tblank, bins)

        if reward_func is not None:
            df.loc[:, 'reward'] = get_reward(df.copy(), reward_func)
            df.loc[:, 'reward_sum'] = np.cumsum(df.reward)

        if policy is not None:
            df.loc[:, 'pi_s_a'] = policy[df.action, df.state_num_i, df.antennae]

        if ctrl == 'IRL' or ctrl == 'KPB':
            df.loc[:, 'state'] = df['digi_tblank'] + df['antennae'] * 16

        # if ctrl == 'ITX':
        #     df.loc[:, 'DS_burst'] = (
        #         df['DS_mean'].var() - df['DS_mean'].mean()) / (
        #             df['DS_mean'].var() + df['DS_mean'].mean())

        # df.loc[:, 'fano'] = np.power(df['DS_mean'].var(),
        #                              2) / df['DS_mean'].mean()

        # whiff = (df['antennae'].to_numpy() > 0).astype('uint8')
        # hit_on = (whiff[:-1] < whiff[1:])
        df.loc[:, 'pulse'] = np.diff(df.wSum, prepend=df.wSum[0])
        # df.loc[:, 'hit_B'] = get_burstiness(df['tblank'], 300)
        # plt.plot(df.Time, df.hit_B)
        # plt.plot(df.Time, df.tblank)

        # Calculate angular velocity
        dTh = np.gradient(theta)
        dTh = ((dTh + np.pi) % (2 * np.pi)) - np.pi
        df['angular_vel'] = dTh / tstep

        # df['traveled_distance'] = get_traveled_distance(df, 'x', 'y')
        df['traveled_distance'] = traveled_distance(df['x'], df['y']).cumsum()
        net_displacement = np.hypot(df['x'].iloc[-1] - df['x'].iloc[0],
                                    df['y'].iloc[-1] - df['y'].iloc[0])

        df['tortuosity'] = df['traveled_distance'] / net_displacement
        df['cdv'] = centerline_deviation(df['y'], tstep)
        # df['heading'] = np.cos(np.pi - df['theta'])
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        df['heading'] = np.cos(np.pi - theta)

        # Calculate linear velocity
        df['linear_vel'] = preproc.get_linear_vel(df['x'], df['y'], tstep)

        if ctrl == 'KPB':
            encode_actions(df, ['linear_vel', 'angular_vel'], 3.9918, -0.156,
                           0.39)

        # Remove outliers in linear and angular velocity
        # z = np.abs(stats.zscore(df))
        # z = np.abs(stats.zscore(df[['linear_vel', 'angular_vel']]))
        # df = df[(z <= 3).all(axis=1)]
        df.loc[:, 'dist2src'] = np.sqrt(df.x**2 + df.y**2)
        df['tblank_k'] = df.loc[:, 'tblank'].shift(-1, fill_value=0)
        df['antennae_k'] = df.loc[:, 'antennae'].shift(-1, fill_value=0)

        # Transform blank duration to log scale to reduce skewness
        tblank = df['tblank'].to_numpy()
        log_tblank = np.log1p(tblank)
        df['log_tblank'] = log_tblank
        df['log_tblank_k'] = df.loc[:, 'log_tblank'].shift(-1, fill_value=0)

        # Add column with experiment name
        experiment_id = os.path.basename(os.path.splitext(c)[0])
        df['experiment'] = experiment_id
        df.loc[:, 'psi'] = _sigmoid(df.wSum*(df.tb_B + .2))
        # if (np.sqrt(df['x_mm'].iloc[-1]**2 + df['y_mm'].iloc[-1]**2) < 50):
        # print('Experiment: {} was successful'.format(experiment_id))

        # if resample_length > 0:

        # df = resample_data(df, tstep, resample_length)

        # if Nrows > 0:
        # df = df.drop(df.index[Nrows:])

        # if (timeout > 0):
        # if df['Time'].iloc[-1] <= timeout: dfs.append(df)

        # print(f"(x: {df['x'].iloc[-1]}, y: {df['y'].iloc[-1]})")
        if ignore_fails:
            if reached_goal(df['x'].iloc[-1], df['y'].iloc[-1], 50):
                dfs.append(df)
                # success_rate.append(1)
                success_rate += 1
                success_time.append(df['Time'].iloc[-1])
                tortuosities.append(df['tortuosity'].iloc[-1])

            # else:
            # success_rate.append(np.nan)


        else:
            dfs.append(df)

        lengths.append(len(df.index))

    # success_rate = pd.Series(success_rate).sum() / len(csvs)
    # success_rate /= len(csvs)
    success_time = pd.Series(success_time)# .mean()
    tortuosities = pd.Series(tortuosities)
    print('  Success rate: {:.4f}'.format(success_rate / len(csvs)))
    print('  Search time: {:.3f} +- {:.3f}'.format(success_time.mean(),
                                                   success_time.std()))
    print('  Tortuosity: {:.3f} +- {:.3f}'.format(tortuosities.mean(),
                                                  tortuosities.std()))

    # print('[--- Success rate: {:.3f} ---]'.format(len(dfs) / len(csvs)))

    if ignore_idx:
        dfs = pd.concat(dfs, axis=0, ignore_index=True)

    else:
        dfs = pd.concat(dfs, axis=0, ignore_index=False)

    sr = [success_rate, (len(csvs) - success_rate)]
    return dfs, sr, success_time, tortuosities


def encode_actions(df, action_cols, lv_min, av_lo, av_hi):
    lin_vel, ang_vel = tuple(action_cols)
    # lv_min = df[lin_vel].mean() - df[lin_vel].std()
    # lv_lo, lv_hi = (self.df[lin_vel].quantile(0.45),
    # self.df[lin_vel].quantile(0.55))
    # av_lo, av_hi = (-self.df[ang_vel].mean(), self.df[ang_vel].mean())
    # av_lo, av_hi = (-.087, .087)
    # av_lo, av_hi = (self.df[ang_vel].quantile(0.40),
    # self.df[ang_vel].quantile(0.60))

    # if verbose:

    # print('Min. linear vel. : {:.5f}'.format(lv_min))
    # print('Linear vel. range: ({:.5f}, {:.5f})'.format(lv_lo, lv_hi))
    # print('Angular vel. range: ({:.5f}, {:.5f})'.format(av_lo, av_hi))

    # stop = (self.df[lin_vel] < lv_min) & (self.df[ang_vel].between(
    #     av_lo, av_hi, inclusive=True))
    surge = df[lin_vel].gt(lv_min) & df[ang_vel].between(
        av_lo, av_hi, inclusive=True)
    turn_ccw = ~(surge) & (df[ang_vel] > av_hi)
    turn_cw = ~(surge) & (df[ang_vel] < av_lo)

    for i, a in enumerate([surge, turn_ccw, turn_cw]):
        df.loc[a, 'action'] = i + 1

    # self.df.loc[stop, 'action'] = 0
    df['action'].fillna(0, inplace=True)
    df['action'] = df.action.astype('uint8')

def get_burstiness(x, N):

    # dts = []
    # for i, g in df.groupby((df.tblank.diff() < 0).cumsum()):
    #     dts.append(g.tblank.iloc[-1])
    # ax.plot(g.x_mm, g.y_mm, linewidth=.2, alpha=0.5, color='k', zorder=3)

    # x = pd.Series(dts)
    B = (x.rolling(N).std() - x.rolling(N).mean()) / (x.rolling(N).std() +
                                                      x.rolling(N).mean())
    return B

def digitize_tblank(tblank, _bins):

    bins = _bins['tblank'].to_numpy()
    Nbins = len(bins) - 1
    rtol = 1.e-5
    atol = 1.e-8
    eps = atol + rtol * tblank
    d = np.digitize(tblank + eps, bins[1:])
    d = np.clip(d, 0, Nbins - 1)

    return d

def get_reward(df, reward_func):

    return reward_func[df.digi_tblank, df.antennae]


def get_expert_demos(df):

    numeric_states = {0: ['log_tblank', 16, True, True, True]}
    # categoric_states = ['antennae', 'wind']
    categoric_states = ['antennae']
    action_cols = ['linear_vel', 'angular_vel']
    encode_actions(df, ['linear_vel', 'angular_vel'], 3.9918, -0.156, 0.39)

    _mdp = mdp.MothMDP(df, numeric_states, categoric_states, action_cols)
    _logger.debug(
        _mdp.df[['Time', 'linear_vel', 'angular_vel', 'tblank',
                 'log_tblank']].describe())

    _mdp.encode_states()
    # _mdp.encode_actions(verbose=False)

    _logger.debug(_mdp.df.columns)

    _logger.debug(_mdp.df.groupby('action').linear_vel.describe())
    _logger.debug(_mdp.df.groupby('action').angular_vel.describe())
    mdp_tp = _mdp.get_transition_probability()
    mdp_edges = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in _mdp.digi_edges.items()]))

    # print(mdp_edges / df.Time.iloc[1])
    print(mdp_edges.T)

    mdp_demos = _mdp.df[[
        'Time', 'x', 'y', 'linear_vel', 'angular_vel', 'tblank',
        'antennae', 'state_num_i', 'state_i', 'action', 'reward'
    ]].copy()
    # print(mdp_demos['hit_rate'].describe())

    _logger.debug('Normalized value counts per action')
    _logger.debug(_mdp.df['action'].value_counts(normalize=True, sort=False))
    # print()
    # plt.plot(_mdp.df['tblank'].unique(),
    #  _mdp.df['tblank'].value_counts(normalize=True, sort=False))
    # sns.histplot(
    #     data=_mdp.df,
    #     x='log_tblank',
    #     #  hue=hue,
    #     #  log_scale=True,
    #     cumulative=True,
    #     element='step',
    #     fill=False,
    #     stat='density',
    #     common_norm=False)

    # plt.show()
    print(_mdp.info)

    return mdp_demos, mdp_edges, mdp_tp


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)


    _logger.info('Starting script')
    if len(args.compare) > 2:

        if args.stat_test:
            compare_experiments_stats(args.compare)

        else:
            dset = compare_experiments(args.compare)

    else:
        # print(type(args.compare[0]))
        dset, _, _, _ = merge_data(args.compare[1],
                               ctrl=args.compare[0],
                               bins=None,
                            #    ignore_idx=False,
                               reward_func=None,
                               ignore_fails=True)

        # sns.lineplot(data=dset, x='Time', y='hit_B')
        # plt.show()

        # dset['antennae'] = dset['antennae'].gt(0).astype('uint8')

    # bins = os.path.join(os.getcwd(), 'bin_edges', 'c1ZN98.csv')
    # bins = os.path.join(os.getcwd(), 'bin_edges', 'c3.csv')
    # bins = pd.read_csv(bins)
    # reward = pd.read_csv(os.path.join(os.getcwd(), 'Reward.csv'),
    #  usecols=[1, 2, 3, 4]).to_numpy()


    # policy = os.path.join(os.getcwd(), 'irl_policies', 'c1.h5')
    # with h5py.File(policy, 'r') as h5:
    # simpol = h5['policy'][:]

    # print(simpol.shape) # shape is actions X Ntblank X Nantennae
    # sys.exit()
    # print(reward)



    # sns.lineplot(data=dset[dset['antennae'].gt(0)], x='Time', y='entropy')
    # sns.lineplot(data=dset, x='Time', y='entropy')
    # sns.lineplot(data=dset, x='Time', y='hitrate')
    # plt.show()
    # plot_entropyrms_map(dset[['Time', 'x', 'y', 'antennae', 'S_rms']].copy())
    # sys.exit()
    # fig, axs = plt.subplots(2, 1, figsize=(7, 4))
    # sns.kdeplot(dset['reward'], ax=axs[0])
    # sns.lineplot(x='Time', y='reward_sum', data=dset, ax=axs[1])
    # axs[1].set_ylim(-500, 600)
    # plt.show()
    # print(dset[['linear_vel', 'angular_vel', 'tblank']].describe())
    # mdp_demos, mdp_edges, mdp_tp = get_expert_demos(dset.copy())
    # print(mdp_demos.action.value_counts(sort=True, normalize=True))

    if args.demos:
        # expert_demos = [i for i in glob.glob(os.path.join(args.demos, '*.csv'))]
        demo_dfs, _ = merge_data(args.demos,
                                 ctrl='MOTH',
                                 bins=bins,
                                 reward_func=reward,
                                 policy=simpol,
                                 ignore_fails=False)

    # mhd = get_mhd(demo_dfs, mdp_demos)
    # mhd = get_revised_mhd(demo_dfs, dset)
    # print('Mean modified hausdorff distance: {}'.format(mhd.mean()))
    # mhd.to_csv('MHD_full_{}.csv'.format(args.controller), index=False)
    # RMSPE = get_RMS_error(expert_demos, mdp_demos, 'tblank')
    # moth_tblank_histogram = demo_dfs.tblank.value_counts(normalize=True)
    # sim_tblank_histogram = dset.tblank.value_counts(normalize=True)
    # nll = negative_log_loss(demo_dfs)
    # print('Mean NLL: {}'.format(nll))

    # print(30 * '#')
    # print(type(moth_tblank_histogram))
    # print(type(sim_tblank_histogram))
    # plt.plot(moth_tblank_histogram)
    # plt.plot(sim_tblank_histogram)
    # plt.show()
    # rmspe = np.sqrt(
    #     np.sum((moth_tblank_histogram.sub(sim_tblank_histogram))**2) /
    #     np.sum(moth_tblank_histogram**2))
    # rmspe = RMSPE(moth_tblank_histogram, sim_tblank_histogram)
    # rmspe = get_reward_RMSPE(demo_dfs, bins, reward, dset)
    # print('Mean RMS percentage error: {}'.format(rmspe.mean()))
    # rmspe.to_csv('RMSPE_tb_{}.csv'.format(args.controller), index=False)

    # print(mdp_demos['antennae'].value_counts(normalize=True, sort=False))
    # print(mdp_demos['state_i'].value_counts(normalize=True, sort=False))
    # tblank_vs_vel = dset.groupby('tblank')[[
    # 'linear_vel', 'angular_vel', 'tblank'
    # ]].mean()
    # print(tblank_vs_vel)
    # print(dset[['linear_vel', 'angular_vel', 'tblank']].describe())
    # print('Normalized value counts per action')
    # print(dset['action'].value_counts(normalize=True, sort=False))
    # print(dset['digi_tblank'].value_counts(normalize=True, sort=True))
    # sns.set_style('ticks')
    # sns.set_context('paper')
    # fig, ax = plt.subplots(figsize=(7, 4))
    # ax = sns.lineplot(data=dset, x='Time', y='tortuosity')
    # ax = sns.histplot(data=dset, x='angular_vel', kde=True, stat='density')
    # fig, axs = plt.subplots(2, 1, figsize=(7, 4))
    # ax = sns.lineplot(data=tblank_vs_vel.to_numpy().T)
    # sns.lineplot(data=dset, x='tblank', y='linear_vel', ax=axs[0])
    # sns.lineplot(data=dset, x='tblank', y='angular_vel', ax=axs[1])
    # ax = sns.violinplot(data=dset.tblank)
    # fig, ax = plt.subplots()#figsize=(3.5, 2))
    # ax = sns.lineplot(x='Time', y='cdv', data=dset)
    # ax.set_xlim(0, 260.)
    # ax.set_xscale('log')
    # ax.set_ylabel(r'$\Delta$ heading (deg)')
    # ax.set_xlabel('Blank duration')
    # ax.set_xlim(0, 109 / 30)
    # ax.set_ylim(-90, 90)
    # ax.set_xlim(0, 2)
    # fig.tight_layout()
    # sns.despine(fig)
    # plt.savefig(os.path.join(args.input_dir,
    #  args.controller + '_tortuosity_all'),
    # dpi=300)
    # plt.show()

    # plotter = mdp_plots.MdpPlots('ticks', 'paper', (3.5, 2.6))

    if args.plot:
        plotter = mdp_plots.MdpPlots('whitegrid', 'paper', (7, 2.6))
        outpath = None
        if len(args.plot) > 1:
            outpath = os.path.join(args.input_dir, args.plot[-1])

        if args.plot == 'reward':
            reward = pd.read_csv(os.path.join(args.input_dir, 'Reward_rev.csv'))
            plotter.plot_reward_function(reward,
                                         'Time since last odor hit (s)',
                                         ['None', 'Right', 'Left', 'Both'],
                                         outpath + '_logx_v2',
                                         _logx=True)

        if args.plot[0] == 'sim-trajs':
            plotter.plot_sim_trajectories(args.compare[1],
                                          xlim=(0, 600),
                                          ylim=(-360, 360),
                                          output=outpath)

        if args.plot[0] == 'cmap-traj':
            plotter.cmap_trajectories(dset,
                                      'psi',
                                      xlim=(0, 600),
                                      ylim=(-360, 360),
                                      cbar_label=r'Switching coefficient $\psi$',
                                      output=outpath)

        if args.plot[0] == 'entropy':
            plotter.plot_entropy(dset, hue='Controller', output=outpath)

        if args.plot[0] == 'burstiness':
            plotter.plot_burstiness(dset, output=outpath)
            #   N=20)

        if args.plot[0] == 'heatmap':
            plotter.plot_heatmap(dset, x='x', y='y', z='burstiness')

        if args.plot[0] == 'onevar':
            plotter.plot_onevar(dset, x='dist2src', y='psi')

        if args.plot[0] == 'xy-kde':
            plotter.kdeplots(dset, ['dist2src', 'entropy'],
                             col='burstiness',
                             output=outpath)
            # plotter.kdeplots(dset, ['entropy', 'fano'],
            #                  col='burstiness',
            #                  output=outpath)

        if args.plot[0] == 'scatter':
            plotter.scatterplot(dset,
                                x='x',
                                y='y',
                                size='burstiness',
                                hue='entropy')

        if args.plot[0] == 'sim-xyjoint':
            plotter.plot_sim_xyjoint(args.input_dir,
                                          xlim=(0, 600),
                                          ylim=(-360, 360),
                                          output=outpath)

        if args.plot[0] == 'policy':
            policy = pd.read_csv(
                os.path.join(args.input_dir, 'policy_for_plot.csv'))
            plotter.plt_policy(policy[policy.antennae == 'none'], 'tblank',
                               'probability', 'antennae', 'action',
                               r'Blank duration $\tau_b$ (s)', 'Probability',
                               outpath + '_test')


        if args.plot[0] == 'meanplume':
            plotter.plot_mean_plume(args.input_dir,
                                    'smokevid',
                                    save_path=outpath + '_smokevid')

        if args.plot[0] == 'varplume':
            plotter.plot_variance_plume(
                args.input_dir,
                'silkmoth_smokevid_IRL_200runs_0122_085954',
                'smokevid',
                save_path=outpath + '_smokevid_v2')

        if args.plot[0] == 'plume-snaps':
            plotter.plot_plume_snapshots(args.input_dir, 'smokevid', save_path=outpath + '_smokevid_snap')


    _logger.info('End of script')
    # hm_reward = pd.read_csv(
    #     os.path.join(args.input_dir, 'Reward_for_heatmap.csv'), index_col='tblank')
    # usecols=[1, 2, 3, 4])

    # sns.set_style('ticks')
    # sns.set_context('paper')
    # fig, ax = plt.subplots(figsize=(7, 4))

    # ax = sns.lineplot(x='tblank',
    #                   y='Reward',
    #                   hue='Antennae',
    #                   style='Antennae',
    #                   linewidth=2,
    #                 #   palette='mako',
    #                   data=reward)
    # ax.set_xscale('log')
    # plt.savefig('Reward-line.png', dpi=300)
    # plt.show()
    # print(reward['Reward'].max())
    # plt.plot()

    # plotter.kinematics(dset,
    #                    'tblank', (0, 25), (-np.pi / 2, np.pi / 2),
    #                    logscale=True)
    # plotter.plot_heatmap(tblank_vs_vel, 'linear_vel', 'angular_vel',
    #  'digi_tblank', 'lvel', 'angvel', 'states',
    #  os.getcwd())

    # print(dset.tail(20))
    # traveled_dist = get_traveled_distance(dset, 'x', 'y')
    # print(traveled_dist)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()