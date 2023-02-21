"""
Generate state-action trajectories plus other useful stats from Shigaki's 
2020 tethered moth experiments which incorporate wind stimuli.
IN: path of directory containing log files (csv format)
OUT: Csv files with state-action trajectories
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing
import mdp
import mdp_plots
import fileIO

sns.set(font="sans-serif", rc={"font.sans-serif": ["DejaVu Sans", "Arial"]})


def parse_args(args):
    """Parse command line parameters
    Args: args ([str]): command line parameters as list of strings
    Returns: obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--save-csv",
        dest="save_csv",
        help='Save merged dataframe to csv',
        nargs='?',
        const='rldemos')
    return parser.parse_args(args)


def get_expert_demos(df):
    numeric_states = {0: ['log_tblank', 16, True, True, True]}
    # categoric_states = ['antennae', 'wind']
    categoric_states = ['antennae']
    action_cols = ['linear_vel', 'angular_vel']

    _mdp = mdp.MothMDP(df, numeric_states, categoric_states, action_cols)
    print("============================================================")
    print(_mdp.df[['linear_vel', 'angular_vel', 'tblank']].describe())

    _mdp.encode_states()
    _mdp.encode_actions()
    # _mdp.encode_many_actions(verbose=True)
    # Min. linear vel. : 3.99180
    # Angular vel. range: (-0.11700, 0.35100)

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
        'Time', 'x_mm', 'y_mm', 'linear_vel', 'angular_vel', 
        'tblank', 'log_tblank', 'lasthit', 'hit_rate', 'wind',
        'antennae', 'state_num_i', 'state_i', 'action'
    ]].copy()
    # print(mdp_demos['hit_rate'].describe())

    # Normalized value counts per action
    print(_mdp.df['action'].value_counts(normalize=True, sort=False))
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
    dfs, lengths = preprocessing.merge_data(args.input_dir, timeout=260)
    mdp_demos, mdp_edges, mdp_tp = get_expert_demos(dfs.copy())

    # sns.histplot(data=mdp_demos, x='angular_vel', kde=True, stat='density')
    # plt.show()

    # features = mdp_demos.groupby('state_i')[['wind', 'hits', 'linear_vel', 'angular_vel']].mean()
    # features = mdp_demos.groupby('state_i')[['wind', 'angular_vel', 'log_twhiff', 'lasthit']].mean()
    features = mdp_demos.groupby('state_i')[['wind', 'angular_vel']].median()
    features['wind'] = features.wind.astype('uint8')

    phi = np.zeros((mdp_tp.shape[0], 2))
    phi[np.array(features.index)] = features.to_numpy()
    features = pd.DataFrame(phi)

    # print('Unique values of hit rate: {}'.format(len(features['hit_rate'].unique())))
    # features['hit_rate'] = features.hit_rate.astype('uint8')
    # features['angular_vel'] = np.sign(features.angular_vel).astype('int')

    print('Shape of feature matrix{}'.format(features.shape))
    out_dir = 'rldemos_{}'.format(fileIO.tstamp())
    out_path = fileIO.make_dir(args.input_dir, out_dir)
    np.save(os.path.join(out_path, 'trans_prob.npy'), mdp_tp)

    edges_path = fileIO.make_dir(args.input_dir, out_dir + '/edges')
    mdp_edges.to_csv(os.path.join(edges_path, 'bin_edges.csv'), index=False)
    features.to_csv(os.path.join(edges_path, 'features.csv'), index=False)
    csv_path = fileIO.make_dir(args.input_dir, out_dir)

    for i, g in mdp_demos.groupby((mdp_demos.Time.diff() < 0).cumsum()):
        g.to_csv(os.path.join(csv_path, '{0}-{1}.csv'.format(len(g.index), i + 1)), index=False)

    # if args.save_excel:
    #     with pd.ExcelWriter(os.path.join(args.input_dir, '{}_stats.xlsx'.format(
    #                 args.save_excel))) as writer:
    #         mdp_demos.describe().to_excel(writer,
    #                                 float_format="%.4f",
    #                                 sheet_name='Description')
    #         mdp_demos.head(100).to_excel(writer,
    #                                float_format="%.4f",
    #                                sheet_name='Head')
    #         mdp_demos['state_i'].value_counts(normalize=True).to_excel(
    #             writer, float_format="%.4f", sheet_name='States')
    #         mdp_demos['action'].value_counts(normalize=True).to_excel(
    #             writer, float_format="%.4f", sheet_name='Actions')

    if args.plot:
        plotter = mdp_plots.MdpPlots('ticks', 'paper', (3.5, 2.6))
        outpath = os.path.join(out_path, 'plots')
        if args.plot == 'trajectories':
            plotter.plot_moth_trajectories(dfs, (0, 600), (-360, 360), (0, 0, 50), output=outpath + '_trajectories')

        elif args.plot == 'xy-joint':
            plotter.plot_moth_xyjoint(dfs, (0, 600), (-360, 360), (0, 0, 50), output=outpath + '_jointplot')

        elif args.plot == 'actions':
            plotter.plot_actions(
                mdp_demos, 
                4, 
                'action', 
                'Linear vel. (mm/s)',
                'Angular vel. (rad/s)', 
                '',
                ['Stop', 'Surge', 'Turn CCW', 'Turn CW'],
                outpath + '_mg')
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
            # plt.savefig('tethered2020-moth-hitrate-v-time', dpi=300)
            plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])