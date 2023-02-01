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
from sklearn.preprocessing import PowerTransformer, RobustScaler, KBinsDiscretizer, MinMaxScaler
from scipy import stats
from pathlib import Path
# import tunnel_bombyx.utils as utils
import argparse
import datetime
import sys
import logging
import numpy as np
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
sns.set(font="sans-serif", rc={"font.sans-serif": ["FreeSans", "Arial"]})
import os
import glob
import json
import preprocessing as preproc
import mdp as mdp
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
    parser.add_argument('--version',
                        action='version',
                        version='tunnel_bombyx {ver}'.format(ver=1))
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

def joint_distribution(df, x, y, hue, _logscale):

    sns.set_style('ticks')
    sns.set_context('paper')

    # fig, ax = plt.subplots(figsize=(3.5, 2))
    g = sns.jointplot(data=df,
                      x=x,
                      y=y,
                      hue=hue,
                      log_scale=_logscale,
                      kind='kde')


def plot_actions(df,
                 hue,
                 xlabel,
                 ylabel,
                 lg_title,
                 lg,
                 save_path,
                 aspect_equal=False):
    """Plot Angular vs Linear velocity

    Args:
        df (pandas.DataFrame): Data frame
    """

    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    n_actions = len(df[hue].unique())
    _palette = sns.color_palette("husl", 4)
    _markers = ['s', '>', 'v', '^']
    if n_actions < 4:
        _palette = _palette[1:]
        _markers = _markers[1:]
    ax = sns.scatterplot(x='linear_vel',
                         y='angular_vel',
                         data=df,
                         hue=hue,
                         style=hue,
                         markers=_markers,
                         s=8,
                         palette=_palette,
                         alpha=0.67)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    l = ax.legend_
    l.set_title(lg_title)

    for i, x in enumerate(lg):
        l.texts[i].set_text(x)

    if aspect_equal:
        ax.set_aspect('equal')

    fig.tight_layout()
    # sns.despine(fig)
    plt.savefig(save_path, dpi=300)
    plt.show()

def ks_test_mismatch(df, x, c, N):

    np.random.seed(7777)
    match = df[c] > 0
    mismatch = df[c] <= 0

    ks_match = np.random.choice(df.loc[match, x], N)
    ks_mismatch = np.random.choice(df.loc[mismatch, x], N)
    ks_stat, pval = stats.ks_2samp(ks_match, ks_mismatch)
    return (ks_stat, pval)

def plot_expected_reward_RMSE(data, x, hue, lg_title, lg, xlabel, save_path):

    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    sns.lineplot(data=data, x=x, y='RMSE', hue=hue)
    l = ax.legend_
    l.set_title(lg_title)
    l.texts[0].set_text(lg[0])
    l.texts[1].set_text(lg[1])
    l._loc = 9
    ax.set_xlabel(xlabel)
    sns.despine(fig)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)

def plot_trajectories(df, config, output=None):

    xlim = tuple(config["xlim"])
    ylim = tuple(config["ylim"])
    srcx, srcy = tuple(config["srcxy"])
    goal_radius = config["goal_radius"]

    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

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
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    fig.tight_layout()
    # ax.set_aspect('equal')

    if output is not None:
        plt.savefig(output, dpi=300)
    plt.show()


def plot_trajectories_with_cmap(df, hue, config, output=None):

    xlim = tuple(config["xlim"])
    ylim = tuple(config["ylim"])
    srcx, srcy = tuple(config["srcxy"])
    goal_radius = config["goal_radius"]

    sns.set_style('ticks')
    sns.set_context('paper')
    # fig, ax = plt.subplots(figsize=(7, 3.2))
    fig, ax = plt.subplots(figsize=(4.5, 2.5))


    norm = plt.Normalize(df[hue].min(), df[hue].max())

    ax.add_artist(
        Circle((srcx, srcy),
               goal_radius,
               color='r',
               fill=False,
               linestyle='--',
               linewidth=1,
               zorder=1))

    for i, g in df.groupby((df.Time.diff() < 0).cumsum()):

        x = g.x_mm
        y = g.y_mm
        z = g[hue].to_numpy()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=3)

        # Set the values used for colormapping
        lc.set_array(z)
        lc.set_linewidth(1)
        lc.set_alpha(0.5)
        line = ax.add_collection(lc)

    ax.scatter(srcx, srcy, marker='*', c='k')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    cbar = fig.colorbar(line)
    cbar.set_label("Entropy")
    fig.tight_layout()
    # ax.set_aspect('equal')

    if output is not None:
        plt.savefig(output, dpi=300)
        # plt.savefig(output, format='svg')

    plt.show()


def scatterplot_pb_match(df, x, y):

    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots()

    ax = sns.scatterplot(data=df,
                         x=x,
                         y=y,
                         hue='match_pb',
                         style='match_pb',
                         size='hit_rate')

    return ax

def get_expert_demos(df):

    numeric_states = {0: ['log_tblank', 32, True, True, True]}
    # categoric_states = ['antennae', 'wind']
    categoric_states = ['antennae']
    action_cols = ['tblank', 'linear_vel', 'angular_vel']

    _mdp = mdp.MothMDP(df, numeric_states, categoric_states, action_cols)

    # _mdp.encode_states()
    # _mdp.encode_actions_MG()
    # _mdp.encode_actions_KZ()
    # _mdp.get_mismatching_expected_reward()
    # _mdp.get_mismatching_actual_reward()
    # _mdp.get_planning_error()
    # print(_mdp.df['EX_DS'].groupby(_mdp.df.match_pb).describe())

    _logger.info(_mdp.df.columns)
    print(_mdp.df[[
        'Time', 'linear_vel', 'angular_vel', 'tblank', 'hitsum', 'entropy',
        'EX_DS', 'DS', 'RMSE'
    ]].describe())

    # mdp_tp = _mdp.get_transition_probability()
    # mdp_edges = pd.DataFrame(
    # dict([(k, pd.Series(v)) for k, v in _mdp.digi_edges.items()]))
    # sns.violinplot(x='match_pb', y='EX_DS', data=_mdp.df)
    # sns.boxplot(data=_mdp.df[['pmat', 'pmis']])
    # fig, ax = plt.subplots(1, 2)

    # print(_mdp.df[['pmat', 'pmis']].describe())

    # sns.boxplot(prob_match, ax=ax[0])
    # sns.boxplot(prob_mismatch, ax=ax[1])

    # plot_expected_reward_RMSE(_mdp.df, 'hitsum', ('EX_DS', 'RMSE'), 'match_pb')

    # plt.plot(prob_match, linestyle='--')
    # plt.plot(prob_mismatch, linestyle='-.')

    # g = sns.jointplot(data=_mdp.df,
    #                   x="EX_DS",
    #                   y="DS",
    #                   hue='match_pb',
    #                   kind='kde')
    #   xlim=(0, _mdp.df['hitsum'].max()),
    #   ylim=(_mdp.df['EX_DS'].min(), 0))
    # g.plot_joint(sns.kdeplot, zorder=0)
    # g.plot_marginals(sns.rugplot, height=-.15, clip_on=False)

    # print('Colors of seaborn palette: {}'.format(sns.color_palette()))

    # g = jointplot_programmed_behavior_match(_mdp.df[_mdp.df.match_pb.ne(1)],
    #                                         "hitsum",
    #                                         "hit_rate",
    #                                         (0, _mdp.df['hitsum'].max()),
    #                                         (0, _mdp.df['hit_rate'].max()),
    #                                         sns.color_palette()[0],
    #                                         alpha=0.1,
    #                                         rug=False)

    # scatter = scatterplot_pb_match(df, 'hitsum', 'DS')

    print(_mdp.df['action_mg'].describe())
    ks_match = np.random.choice(_mdp.df[_mdp.df.match_pb.eq(1)].DS, 1000)
    ks_mismatch = np.random.choice(_mdp.df[_mdp.df.match_pb.ne(1)].DS, 1000)

    # ks_mismatch = np.random.choice(_mdp.df.match_pb.ne(1), 500)
    # ks_match = np.random.choice(_mdp.df.pmat, 500)
    # ks_mismatch = np.random.choice(_mdp.df.pmis, 500)
    ks_stat, pval = stats.ks_2samp(ks_match, ks_mismatch)
    print(f'KS stat: {ks_stat}, pval: {pval}')

    print(_mdp.df['match_pb'].value_counts(normalize=True))
    print(_mdp.df[['action_kz', 'action_mg']].value_counts(normalize=True))
    plt.show()
    mdp_demos = _mdp.df[[
        'Time', 'linear_vel', 'angular_vel', 'action_mg', 'action_kz'
    ]].copy()

    # print(_mdp.info)

    return mdp_demos

def plot_histogram(df,
                   x,
                   xlabel,
                   lg_title,
                   lg,
                   save_path,
                   hue=None,
                   logx=False):

    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax = sns.histplot(data=df, x=x, hue=hue, log_scale=logx, element='step')
    # ax.legend(loc='upper left')
    # ax._legend.set_title(lg_title)
    # for t, l in zip(ax._legend.texts, lg):
    #     t.set_text(l)
    l = ax.legend_
    l.set_title(lg_title)
    l.texts[0].set_text(lg[0])
    l.texts[1].set_text(lg[1])
    l._loc = 2
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    # plt.savefig(save_path, dpi=300)

def plot_entropy(df, x, xlabel, save_path):

    sns.set_style('ticks')
    sns.set_context('paper')
    # fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    # fig, ax = plt.subplots()
    # fig.set_figwidth(3.5)

    for i, g in df.groupby((df.Time.diff() < 0).cumsum()):
        ax.plot(g.Time, g.entropy, linewidth=1, alpha=0.3, color='k', zorder=3)

    ax = sns.lineplot(data=df, x=x, y='entropy')
    # ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Entropy')
    sns.despine(fig)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)


def plot_states(df, x, y):

    sns.set_style('ticks')
    # fig, ax = plt.subplots()
    ax = sns.jointplot(data=df, x=x, y=y, kind='kde', log_scale=[True, False])


def plot_CDF(df, x, xlabel, lg_title, lg, save_path, hue=None, logx=False):

    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax = sns.histplot(data=df,
                      x=x,
                      hue=hue,
                      log_scale=logx,
                      cumulative=True,
                      element='step',
                      fill=False,
                      stat='density',
                      common_norm=False)

    # ax.legend(loc='upper left')
    l = ax.legend_
    l.set_title(lg_title)
    l.texts[0].set_text(lg[0])
    l.texts[1].set_text(lg[1])
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    # plt.savefig(save_path, dpi=300)

def plot_heatmap(df, x, y, z, xlabel, ylabel, title, save_path):

    # matplotlib.rcParams['text.usetex'] = True
    sns.set_style('ticks')
    sns.set_context('paper')
    # yticks = df[y].unique().astype('uint8')[1:]
    # ytickLabels = map(str, yticks)
    # fig, ax = plt.subplots(figsize=(7, 4))
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax = sns.heatmap(df.pivot_table(z, y, x, fill_value=0), center=0)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(ytickLabels)
    ax.invert_yaxis()
    # ax.tick_params(axis='y', which='major', labelsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.savefig('{}.svg'.format(save_path))
    # plt.show()


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    config_file = os.path.join(args.input_dir, 'config.json')

    with open(config_file) as cf:
        conf = json.load(cf)

    _logger.info("Starting script")

    _logger.info('Reading and merging csv files from specified path...')
    dfs, lengths = preproc.merge_data(args.input_dir, ignore_idx=False)

    # print(dfs.describe())

    # mdp_demos = get_expert_demos(dfs.copy())
    print(dfs[[
        'Time', 'linear_vel', 'angular_vel', 'tblank', 'hitsum', 'entropy',
        'EX_DS', 'DS', 'RMSE'
    ]].describe())

    ks_stat, pval = ks_test_mismatch(dfs, 'DS', 'match_pb', 1000)
    print(f'KS stat: {ks_stat}, pval: {pval}')

    print(dfs['match_pb'].value_counts(normalize=True))
    print(dfs[['action_kz', 'action_mg']].value_counts(normalize=True))
    plt.show()
    mdp_demos = dfs[[
        'Time', 'linear_vel', 'angular_vel', 'action_mg', 'action_kz'
    ]].copy()

    out_dir = '{}_{}'.format(args.save_csv, fileIO.tstamp())

    if args.save_trans_prob:

        out_path = fileIO.make_dir(args.input_dir, out_dir)
        np.save(os.path.join(out_path, '{}.npy'.format(args.save_trans_prob)),
                mdp_tp)

    if args.save_csv:
        edges_path = fileIO.make_dir(args.input_dir, out_dir + '/edges')
        mdp_edges.to_csv(os.path.join(edges_path, 'bin_edges.csv'),
                         index=False)

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
            # dfs['state_i'].value_counts(normalize=True).to_excel(
            # writer, float_format="%.4f", sheet_name='States')
            mdp_demos[['action_kz', 'action_mg']].value_counts(normalize=True).to_excel(
                writer, float_format="%.4f", sheet_name='Actions')

    if args.plot:
        # sns.set_context('paper')
        # out_path = os.path.join(args.input)
        outpath = os.path.join(args.input_dir, args.plot)

        if args.plot == 'trajectories':
            plot_trajectories(dfs, conf["fukushima"], outpath)
            # plot_trajectories(dfs, conf["fukushima"])

        elif args.plot == 'gradient-trajectories':

            plot_trajectories_with_cmap(dfs, 'entropy', conf['fukushima'], outpath)

        elif args.plot == 'states':

            plot_states(dfs, 'hitsum', 'hit_rate')

        elif args.plot == 'entropy':

            plot_entropy(dfs, 'Time', 'Time (s)', outpath + '_time')
            plot_entropy(dfs, 'hitsum', 'Cumulative sum of hits',
                         outpath + '_hitsum')

        elif args.plot == 'actions':
            # plot_actions(mdp_demos, mdp_tp.shape[1], outpath)
            plot_actions(dfs, 'action_mg', 'Linear vel. (mm/s)',
                         'Angular vel. (rad/s)', '',
                         ['Stop', 'Surge', 'Turn CW', 'Turn CCW'],
                         outpath + '_mg')
            plot_actions(dfs, 'action_kz', 'Linear vel. (mm/s)',
                         'Angular vel. (rad/s)', '',
                         ['Surge', 'Turn CW', 'Turn CCW'], outpath + '_kz')

        elif args.plot == 'heatmaps':
            # print(dfs.hit_rate.unique().astype('uint8')[1:])
            plot_heatmap(dfs[dfs.match_pb <= 0], 'hitsum', 'hit_rate', 'DS',
                         'Cumulative sum of hits', 'Hit rate (Hz)',
                         r'Mismatch, $\Delta S_t$', outpath + '_mis_DS_v2')
            plot_heatmap(
                dfs[dfs.match_pb <= 0], 'hitsum', 'hit_rate', 'EX_DS',
                'Cumulative sum of hits', 'Hit rate (Hz)',
                r'Mismatch, $E[\Delta S (\mathbf{r}_t \mapsto \mathbf{r}^\prime)]$',
                outpath + '_mis_EXDS_v2')
            plot_heatmap(dfs[dfs.match_pb > 0], 'hitsum', 'hit_rate', 'DS',
                         'Cumulative sum of hits', 'Hit rate (Hz)',
                         r'Match, $\Delta S_t$', outpath + '_mat_DS_v2')
            plot_heatmap(
                dfs[dfs.match_pb > 0], 'hitsum', 'hit_rate', 'EX_DS',
                'Cumulative sum of hits', 'Hit rate (Hz)',
                r'Match, $E[\Delta S (\mathbf{r}_t \mapsto \mathbf{r}^\prime)]$',
                outpath + '_mat_EXDS_v2')

        elif args.plot == 'jointdist':

            # joint_distribution(dfs[dfs.match_pb <= 0])
            # joint_distribution(dfs[dfs.match_pb > 0])
            joint_distribution(dfs, 'DS', 'hit_rate', 'match_pb', [True, False])

        elif args.plot == 'RMSE':

            plot_expected_reward_RMSE(dfs, 'hitsum', 'match_pb', 'Match',
                                      ['No', 'Yes'], 'Cumulative sum of hits',
                                      outpath + '_hitsum')
            plot_expected_reward_RMSE(dfs, 'hit_rate', 'match_pb', 'Match',
                                      ['No', 'Yes'], 'Hit rate (Hz)',
                                      outpath + '_hitrate')

        elif args.plot == 'histogram':
            plot_histogram(dfs,
                           'DS',
                           r'$\Delta S$',
                           'Match',
                           ['No', 'Yes'],
                           outpath + '_counts_DS',
                           hue='match_pb',
                           logx=True)
            plot_histogram(dfs,
                           'EX_DS',
                           r'$E[\Delta S]$',
                           'Match',
                           ['No', 'Yes'],
                           outpath + '_counts_EXDS',
                           hue='match_pb',
                           logx=False)
            plot_CDF(dfs,
                     'DS',
                     r'$\Delta S$',
                     'Match',
                     ['No', 'Yes'],
                     outpath + '_CDF_DS',
                     hue='match_pb',
                     logx=True)
            plot_CDF(dfs,
                     'EX_DS',
                     r'$E[\Delta S]$',
                     'Match',
                     ['No', 'Yes'],
                     outpath + '_CDF_EXDS',
                     hue='match_pb',
                     logx=False)

        plt.show()

    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
