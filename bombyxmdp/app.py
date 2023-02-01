#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate state-action trajectories plus other useful stats from Shigaki's 2020 tethered moth experiments which incorporate wind stimuli.
IN: path of directory containing log files (csv format)
OUT: Csv files with state-action trajectories
"""

from tqdm import tqdm
from scipy import stats
from os.path import isfile, join
from sklearn.preprocessing import PowerTransformer, RobustScaler, KBinsDiscretizer
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
# import tunnel_bombyx.utils as utils
import argparse
import datetime
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
sns.set(font="sans-serif", rc={"font.sans-serif": ["DejaVu Sans", "Arial"]})
import os
import preprocessing as preproc
import mdp
import mdp_plots
import fileIO

_logger = logging.getLogger(__name__)

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
    parser.add_argument("-p",
                        "--plot",
                        dest="plot",
                        help='Plot reward and action-value function',
                        nargs='?',
                        const='plot.png')
    parser.add_argument("-i",
                        "--input-dir",
                        type=str,
                        dest="input_dir",
                        help='Path of the directory with the input data',
                        required=True)
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

def plot_actions(df, n_actions, save_path, aspect_equal=False):
    """Plot Angular vs Linear velocity

    Args:
        df (pandas.DataFrame): Data frame
    """

    fig, ax = plt.subplots()
    ax = sns.scatterplot(x='angular_vel',
                         y='linear_vel',
                         data=df,
                         hue='action',
                         palette=sns.color_palette("husl", n_actions),
                         alpha=0.67)

    if aspect_equal:
        ax.set_aspect('equal')

    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_trajectories(df, config, output=None):

    xlim = tuple(config["xlim"])
    ylim = tuple(config["ylim"])
    srcx, srcy = tuple(config["srcxy"])
    goal_radius = config["goal_radius"]

    fig, ax = plt.subplots()

    ax.add_artist(
        Circle((srcx, srcy),
               goal_radius,
               color='r',
               fill=False,
               linestyle='--',
               linewidth=0.5,
               zorder=1))

    for i, g in df.groupby((df.Time.diff() < 0).cumsum()):
        ax.plot(g.x_mm, g.y_mm, linewidth=1, alpha=0.3, color='k', zorder=3)

    ax.scatter(srcx, srcy, marker='*', c='k')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    if output is not None:
        plt.savefig(output, dpi=300)
    plt.show()


def plot_blanks(df, output=None):

    # xlim = tuple(config["xlim"])
    # ylim = tuple(config["ylim"])
    # srcx, srcy = tuple(config["srcxy"])
    # goal_radius = config["goal_radius"]

    # fig, ax = plt.subplots()
    x = []
    lens = []

    # ax.add_artist(
    #     Circle((srcx, srcy),
    #            goal_radius,
    #            color='r',
    #            fill=False,
    #            linestyle='--',
    #            linewidth=0.5,
    #            zorder=1))

    for i, g in df.groupby((df.tblank.diff() < 0).cumsum()):
        # ax.plot(g.Time, g.tblank, linewidth=1, alpha=0.3, color='k', zorder=3)
        if len(g) % 33 < 3:
            g.to_csv(os.path.join(os.getcwd(), '{}_{}.csv'.format(i, len(g))), index=False)

        x.append(g.tblank.iloc[-1])
        lens.append(len(g))

    # ax = sns.violinplot(data=x)
    print(pd.Series(x).mean())
    print(pd.Series(lens).median())
    print(g)

    # ax.scatter(srcx, srcy, marker='*', c='k')
    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim)
    # ax.set_aspect('equal')

    if output is not None:
        plt.savefig(output, dpi=300)
    plt.show()


def get_expert_demos(df):

    numeric_states = {0: ['log_tblank', 16, True, True, True]}
    # categoric_states = ['antennae', 'wind']
    categoric_states = ['antennae']
    action_cols = ['linear_vel', 'angular_vel']

    _mdp = mdp.MothMDP(df, numeric_states, categoric_states, action_cols)
    print(_mdp.df[['linear_vel', 'angular_vel', 'tblank']].describe())

    _mdp.encode_states()
    _mdp.encode_actions(verbose=True)
    # _mdp.encode_many_actions(verbose=True)
    _logger.debug(_mdp.df.columns)
    # Min. linear vel. : 3.99180
    # Angular vel. range: (-0.11700, 0.35100)

    _logger.debug(_mdp.df.groupby('action').linear_vel.describe())
    _logger.debug(_mdp.df.groupby('action').angular_vel.describe())
    mdp_tp = _mdp.get_transition_probability()
    mdp_edges = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in _mdp.digi_edges.items()]))

    # print(mdp_edges / df.Time.iloc[1])
    print(mdp_edges.T)
    # for i in np.linspace(0.06, .99, 17):
    # print(_mdp.df.log_tblank.quantile(i))

    # fig, ax = plt.subplots(figsize=(16,9))
    # ax = sns.histplot(_mdp.df.log_tblank, bins=16)
    # ax.plot(np.log1p(mdp_edges))
    # plt.show()
    # mdp_demos = _mdp.df[[
    #     'Time', 'linear_vel', 'angular_vel', 'state_i', 'action'
    # ]].copy()

    mdp_demos = _mdp.df[[
        'Time', 'x_mm', 'y_mm', 'linear_vel', 'angular_vel', 'tblank', 'log_tblank', 'lasthit', 'hit_rate', 'wind',
        'antennae', 'state_num_i', 'state_i', 'action'
    ]].copy()
    # print(mdp_demos['hit_rate'].describe())

    _logger.debug('Normalized value counts per action')
    _logger.debug(_mdp.df['action'].value_counts(normalize=True, sort=False))
    # print()
    # plt.plot(_mdp.df['tblank'].unique(),
    #  _mdp.df['tblank'].value_counts(normalize=True, sort=False))
    # sns.histplot(data=_mdp.df[['log_tblank', 'state_num_i']],
    #  x='tblank',
    #  hue=hue,
    #  log_scale=False,
    #  cumulative=True,
    #  element='step',
    #  fill=True,
    #  bins=32,
    #  stat='probability')
    #  common_norm=True)

    # # norm_cdf = stats.norm.cdf(_mdp.df.log_tblank)

    # data = _mdp.df['log_tblank'].to_numpy()
    # data_sorted = np.sort(data)
    # norm_cdf = 1. * np.arange(len(data)) / (len(data) - 1)
    # plt.plot(data_sorted, norm_cdf)
    # # smooth
    # smooth = gaussian_filter1d(norm_cdf, 100)

    # # compute second derivative
    # smooth_d2 = np.gradient(np.gradient(smooth))

    # # find switching points
    # infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    # print("INflection points")
    # print(infls)

    # plt.scatter(mdp_edges.to_numpy()[:,0], np.zeros(len(mdp_edges)))
    # plt.show()

    print(_mdp.info)

    # print(_mdp.df['tortuosity'].describe())

    # sns.heatmap(mdp_tp[:, 1, :])
    # sns.set_style('ticks')
    # sns.set_context('paper')
    # fig, ax = plt.subplots(figsize=(7, 4))
    # ax = sns.violinplot(data=_mdp.df.tblank)
    # ax = sns.lineplot(x='Time', y='tortuosity', data=_mdp.df)
    # ax.set_xlim(0, 260.)
    # fig.tight_layout()
    # sns.despine(fig)
    # plt.savefig('tethered2020-moth-time-v-tortuosity', dpi=300)
    # plt.show()

    # sns.lineplot(x='Time', y='hit_rate', data=mdp_demos)
    # plt.show()
    # print(_mdp.df.loc[_mdp.df.state_num_i.eq(7), 'tblank'].head(20))
    # fig, axs = plt.subplots(2,1)
    # sns.histplot(_mdp.df.state_num_i, ax=axs[0], bins=32, kde=True)
    # sns.histplot(_mdp.df.log_tblank, ax=axs[1], bins=32, kde=True)
    # plt.show()

    return mdp_demos, mdp_edges, mdp_tp


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    dfs, lengths = preproc.merge_data(args.input_dir, timeout=260)
    # print(dfs[['Time', 'whiff', 'twhiff', 'tblank', 'antennae',
    #    'lasthit']].head(50))
    # dfs, lengths = preproc.merge_data(args.input_dir)

    mdp_demos, mdp_edges, mdp_tp = get_expert_demos(dfs.copy())
    # sns.histplot(data=mdp_demos, x='angular_vel', kde=True, stat='density')
    # plt.show()
    # dflen = len(mdp_demos)
    # # features = mdp_demos.groupby('state_i')[['wind', 'hits', 'linear_vel', 'angular_vel']].mean()
    features = mdp_demos.groupby('state_i')[['wind', 'angular_vel']].median()
    # features = mdp_demos.groupby('state_i')[[
    #     'wind', 'angular_vel', 'log_twhiff', 'lasthit'
    # ]].mean()
    features['wind'] = features.wind.astype('uint8')
    phi = np.zeros((mdp_tp.shape[0], 2))
    # phi = np.zeros((mdp_tp.shape[0], 4))
    phi[np.array(features.index)] = features.to_numpy()

    features = pd.DataFrame(phi)
    # print('Unique values of hit rate: {}'.format(len(features['hit_rate'].unique())))
    # features['hit_rate'] = features.hit_rate.astype('uint8')
    # features['angular_vel'] = np.sign(features.angular_vel).astype('int')
    print('Shape of feature matrix{}'.format(features.shape))
    _logger.debug(features.describe())

    out_dir = '{}_{}'.format(args.save_csv, fileIO.tstamp())

    if args.save_trans_prob:

        out_path = fileIO.make_dir(args.input_dir, out_dir)
        np.save(os.path.join(out_path, '{}.npy'.format(args.save_trans_prob)),
                mdp_tp)

    if args.save_csv:
        edges_path = fileIO.make_dir(args.input_dir, out_dir + '/edges')
        # feats_path = fileIO.make_dir(args.input_dir, out_dir + '/features')

        mdp_edges.to_csv(os.path.join(edges_path, 'bin_edges.csv'),
                         index=False)
        features.to_csv(os.path.join(edges_path, 'features.csv'), index=False)

        csv_path = fileIO.make_dir(args.input_dir, out_dir)

        for i, g in mdp_demos.groupby((mdp_demos.Time.diff() < 0).cumsum()):

            g.to_csv(os.path.join(csv_path,
                                  '{0}-{1}.csv'.format(len(g.index), i + 1)),
                     index=False)

    if args.save_excel:
        with pd.ExcelWriter(
                os.path.join(args.input_dir, '{}_stats.xlsx'.format(
                    args.save_excel))) as writer:
            mdp_demos.describe().to_excel(writer,
                                    float_format="%.4f",
                                    sheet_name='Description')
            mdp_demos.head(100).to_excel(writer,
                                   float_format="%.4f",
                                   sheet_name='Head')
            mdp_demos['state_i'].value_counts(normalize=True).to_excel(
                writer, float_format="%.4f", sheet_name='States')
            mdp_demos['action'].value_counts(normalize=True).to_excel(
                writer, float_format="%.4f", sheet_name='Actions')

    if args.plot:
        plotter = mdp_plots.MdpPlots('ticks', 'paper', (3.5, 2.6))
        # out_path = os.path.join(args.input)
        outpath = os.path.join(args.input_dir, args.plot)
        if args.plot == 'trajectories':
            # plot_trajectories(dfs, conf["mothVR"], outpath)
            plotter.plot_moth_trajectories(dfs, (0, 600), (-360, 360),
                                           (0, 0, 50), output=outpath + '_moth')

        elif args.plot == 'xy-joint':
            # plot_trajectories(dfs, conf["mothVR"], outpath)
            plotter.plot_moth_xyjoint(dfs, (0, 600), (-360, 360), (0, 0, 50),
                                      output=outpath + '_jointplot')

        elif args.plot == 'blanks':
            # plot_trajectories(dfs, conf["mothVR"], outpath)
            plot_blanks(dfs)

        elif args.plot == 'actions':
            # plot_actions(mdp_demos, mdp_tp.shape[1], outpath)
            # plotter.plot_actions(mdp_demos, 4, 'action', 'Linear vel. (mm/s)',
            #                      'Angular vel. (rad/s)', '',
            #                      ['Stop', 'Surge', 'Turn CCW', 'Turn CW'],
            #                      outpath + '_mg')
            plotter.plt_action_histograms(
                mdp_demos.copy(),
                'linear_vel',
                'action',
                'Linear vel (mm/s)',
                'Action', ['Stop', 'Surge', 'Turn CCW', 'Turn CW'],
                bins=32,
                save_path=outpath + '_linv_v2')
            plotter.plt_action_histograms(
                mdp_demos.copy(),
                'angular_vel',
                'action',
                'Angular vel (rad/s)',
                'Action', ['Stop', 'Surge', 'Turn CCW', 'Turn CW'],
                binrange=[-2*np.pi, 2*np.pi],
                save_path=outpath + '_angv_v2')

        elif args.plot == 'heatmap':

            plotter.plot_heatmap(mdp_demos, 'state_num_i', 'antennae',
                                 'hits', 'Blank duration',
                                 'Hit antenna', 'Cumulative hits',
                                 outpath + '_mis_DS')

            plotter.plot_heatmap(mdp_demos, 'state_num_i', 'antennae', 'wind',
                                 'Blank duration', 'Hit antenna',
                                 'Wind direction', outpath + '_mis_DS')

        elif args.plot == 'kinematics':
            plotter.kinematics(dfs,
                               'tblank', (0, 25), (-np.pi / 2, np.pi / 2),
                               logscale=True)

        elif args.plot == 'states':
            plotter.plot_states(mdp_demos.copy(), 'log_tblank', 'antennae',
                                r'$\log(1+\tau_b)$', 'Hit antennae',
                                ['None', 'Right', 'Left', 'Both'], save_path=outpath + '_logtb_v2')
            plotter.plot_states(mdp_demos.copy(), 'state_num_i', 'antennae',
                                r'Discretized $\tau_b$', 'Hit antennae',
                                ['None', 'Right', 'Left', 'Both'], save_path=outpath + '_disc_v2')

        elif args.plot == 'heading':
            sns.set_style('ticks')
            sns.set_context('paper')
            # fig, ax = plt.subplots(figsize=(3.5, 2))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax = sns.lineplot(x='tblank', y='heading', data=dfs)
            # ax.set_xscale('log')
            # ax.set_ylabel(r'$\cos(\pi - \theta_{moth})$')
            ax.set_ylabel(r'$\Delta$ heading (deg)')
            # ax.set_xlabel('Blank duration')
            ax.set_xlabel('tblank')
            fig.tight_layout()
            sns.despine(fig)
            # ax.set_xlim(0, 2)
            ax.set_xlim(0, 109 / 30)
            # ax.set_ylim(-90, 90)
            plt.savefig('tethered2020-moth-heading-v-tblank', dpi=300)
            plt.show()

        elif args.plot == 'hit-rate':
            sns.set_style('ticks')
            sns.set_context('paper')
            fig, ax = plt.subplots(figsize=(3.5, 2))
            ax = sns.lineplot(x='Time', y='hit_rate', data=dfs)
            # ax.set_xscale('log')
            ax.set_ylabel('Hit rate')
            # ax.set_xlabel('Blank duration')
            ax.set_xlabel('Time')
            fig.tight_layout()
            sns.despine(fig)
            # ax.set_xlim(0, 2)
            plt.savefig('tethered2020-moth-hitrate-v-time', dpi=300)
            plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])