import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set(font="sans-serif", rc={"font.sans-serif": ["FreeSans", "Arial"]})
import numpy as np
import pandas as pd
import glob
import os
import h5py
from tqdm import tqdm

class MdpPlots(object):
    def __init__(self, style, context, figsize, dpi=300, tight=True):
        self.style = style
        self.context = context
        self.figsize = figsize
        self.dpi = dpi
        self.tight = tight

    

    def plot_sim_xyjoint(self, _path, xlim, ylim, src=(0, 0, 50), N=0, output=None):

        # srcx, srcy, goal_radius = src

        sns.set_style('ticks')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=(3.4, 2.83))

        csvs = [i for i in glob.glob(os.path.join(_path, '*.csv'))]
        if N > 0:
            csvs = np.random.choice(csvs, N)

        dfs = []
        for c in csvs:
            # df = pd.read_csv(c, usecols=[2, 3])
            df = pd.read_csv(c, usecols=['x_mm', 'y_mm'])
            dfs.append(df)

        dfs = pd.concat(dfs, axis=0, ignore_index=True)

        ax = sns.jointplot(x='x_mm',
                           y='y_mm',
                           data=dfs,
                           kind='kde',
                           xlim=(0, 600),
                           ylim=(-360, 360))
        #    marginal_kws=dict(bins=32))
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    def plot_burstiness(self, dset, output=None):
        sns.set_style('ticks')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        ax = sns.lineplot(data=dset, x='Time', y='tb_B')
        ax = sns.lineplot(data=dset, x='Time', y='pulse')
        ax.fill_between(dset.Time, 0, dset.pulse, alpha=0.3)
        # csvs = [i for i in glob.glob(os.path.join(_path, '*.csv'))]

        # if N > 0:csvs = np.random.choice(csvs, N)

        # for c in csvs:
        #     # df = pd.read_csv(c, usecols=[2, 3])
        #     df = pd.read_csv(c, usecols=['Time', 'entropy'])
        #     ax.plot(df.Time,
        #             df.entropy,
        #             linewidth=1,
        #             alpha=0.3,
        #             color='black',
        #             zorder=3)

        ax.set_xlabel('Time')
        ax.set_ylabel('Burstiness')
        sns.despine(fig)
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    def plot_entropy(self, dset, hue=None, N=0, output=None):
        sns.set_style('ticks')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        ax = sns.lineplot(data=dset, x='Time', y='entropy', hue=hue)
        # csvs = [i for i in glob.glob(os.path.join(_path, '*.csv'))]

        # if N > 0:csvs = np.random.choice(csvs, N)

        # for c in csvs:
        #     # df = pd.read_csv(c, usecols=[2, 3])
        #     df = pd.read_csv(c, usecols=['Time', 'entropy'])
        #     ax.plot(df.Time,
        #             df.entropy,
        #             linewidth=1,
        #             alpha=0.3,
        #             color='black',
        #             zorder=3)

        ax.set_xlabel('Time')
        ax.set_ylabel('Entropy')
        sns.despine(fig)
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    def scatterplot(self, dset, x, y, size=None, hue=None, N=0, output=None):
        sns.set_style('ticks')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        ax = sns.scatterplot(data=dset, x=x, y=y, size=size, hue=hue, sizes=(1, 100))
        # csvs = [i for i in glob.glob(os.path.join(_path, '*.csv'))]

        # if N > 0:csvs = np.random.choice(csvs, N)

        # for c in csvs:
        #     # df = pd.read_csv(c, usecols=[2, 3])
        #     df = pd.read_csv(c, usecols=['Time', 'entropy'])
        #     ax.plot(df.Time,
        #             df.entropy,
        #             linewidth=1,
        #             alpha=0.3,
        #             color='black',
        #             zorder=3)

        ax.set_xlabel('Time')
        ax.set_ylabel('Entropy')
        sns.despine(fig)
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    def plot_onevar(self, df, x, y, _figsize=(3.4, 2.5), hue=None,
                    output=None):
        sns.set_style('ticks')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=_figsize)
        ax = sns.lineplot(data=df, x=x, y=y, hue=hue)

        # ax.set_xlabel('Time')
        # ax.set_ylabel('Entropy')
        sns.despine(fig)
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()


    def cmap_trajectories(self,
                          df,
                          hue,
                          xlim,
                          ylim,
                          cbar_label,
                          N=0,
                          src=(0, 0, 50),
                          output=None):

        srcx, srcy, goal_radius = src
        # df = df[df[hue].between(-.1, 0, inclusive=False)]
        # df = df[(df[hue].ge(7))
        # & (df['tb_B'].between(-.3, 0, inclusive=True))]
        # df = df[(df['hit_B'].lt(.5)) & (df['tb_B'].lt(-.2))]
        # df = df[(df['entropy'].between(3, 5, inclusive=True))
        # & (df['tb_B'].between(-.2, -.1, inclusive=True))]

        sns.set_style('dark')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=(3.4, 2.83))
        norm = plt.Normalize(df[hue].min(), df[hue].max())

        ax.add_artist(
            Circle((srcx, srcy),
                   goal_radius,
                   color='r',
                   fill=False,
                   linestyle='--',
                   linewidth=0.5,
                   zorder=1))

        for i, g in df.groupby((df.Time.diff() < 0).cumsum()):

            x = g.x
            y = g.y
            z = g[hue].to_numpy()

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap='viridis_r', norm=norm, zorder=3)

            # Set the values used for colormapping
            lc.set_array(z)
            lc.set_linewidth(1)
            lc.set_alpha(0.5)
            line = ax.add_collection(lc)

        ax.scatter(srcx, srcy, marker='*', c='k')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        cbar = fig.colorbar(line)
        cbar.set_label(cbar_label)
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()



    def plot_sim_trajectories(self,
                               _path,
                               xlim,
                               ylim,
                               N=0,
                               src=(0, 0, 50),
                               output=None):

        # xlim = tuple(config["xlim"])
        # ylim = tuple(config["ylim"])
        srcx, srcy, goal_radius = src

        sns.set_style('ticks')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=(3.4, 2.83))
        csvs = [i for i in glob.glob(os.path.join(_path, '*.csv'))]

        if N > 0:
            csvs = np.random.choice(csvs, N)

        ax.add_artist(
            Circle((srcx, srcy),
                   goal_radius,
                   color='r',
                   fill=False,
                   linestyle='--',
                   linewidth=0.5,
                   zorder=1))

        for c in csvs:
            # df = pd.read_csv(c, usecols=[2, 3])
            # df = pd.read_csv(c, usecols=['x_mm', 'y_mm'])
            df = pd.read_csv(c, usecols=['x_mm', 'y_mm', 'mode'])
            itx = df[df['mode'] == 'ITX'].copy()
            kpb = df[df['mode'] == 'KPB'].copy()
            ax.plot(itx.x_mm,
                    itx.y_mm,
                    # linewidth=.5,
                    alpha=0.5,
                    color='black',
                    zorder=3)

            ax.plot(kpb.x_mm,
                    kpb.y_mm,
                    # linewidth=.5,
                    alpha=0.5,
                    color='#FF00FF',
                    zorder=3)

        ax.scatter(srcx, srcy, marker='*', c='k')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    def plot_states(self,
                    df,
                    x,
                    hue,
                    xlabel,
                    lg_title,
                    lg,
                    logx=False,
                    save_path=None):

        num2cat = {0: 'None', 1: 'Right', 2: 'Left', 3: 'Both'}
        df.loc[:, hue] = df[hue].map(num2cat)

        sns.set_style('ticks')
        sns.set_context('paper')

        fig, ax = plt.subplots(figsize=(3.4, 2), sharey=True)

        ax = sns.histplot(data=df,
                          x=x,
                          hue=hue,
                          element='step',
                        #   stat='probability',
                          fill=False,
                          bins=16,
                          palette='Dark2',
                          log_scale=logx)

        ax.set_xlabel(xlabel)

        l = ax.legend_
        l.set_title(lg_title)

        # for j, x in enumerate(lg):
        # l.texts[j].set_text(x)

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.savefig(save_path + '.svg')
        plt.show()

    
    def plt_policy(self, df, x, y, col, cat, xlabel, ylabel, save_path=None):

        sns.set_style('ticks')
        sns.set_context('paper')
        fig, ax = plt.subplots(figsize=(7, 2.6))

        ax = sns.scatterplot(data=df, x=x, y=y, hue=cat, style=cat)
        # ax.set_xscale('log')
        fig.tight_layout()

        # ax = sns.relplot(data=df, x=x, y=y, col=col, hue=cat, style=cat, kind='scatter', height=2, aspect=2, col_wrap=1)

        # (ax.set_axis_labels(xlabel, ylabel).set_titles(
        #     "Hit antennae: {col_name}").tight_layout(w_pad=0))

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.savefig(save_path + '.svg')
        plt.show()

    def plot_mean_plume(self,
                        _path,
                        env,
                        extx=(0, 600),
                        exty=(-360, 360),
                        save_path=None):

        vids = [i for i in glob.glob(os.path.join(_path, env, '*.h5'))]

        means = []

        for v in vids:
            with h5py.File(v, 'r') as h5:
                mat = h5['frames'][:]
                means.append(np.mean(mat, axis=0))

        mean = np.mean(np.array(means), axis=0)

        sns.set_style('white')
        sns.set_context(self.context)
        ext = extx + exty
        fig, ax = plt.subplots()
        img = ax.imshow(mean, extent=ext, cmap='plasma')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(img, cax=cax)
        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.savefig(save_path + '.svg')

        plt.show()

    def plot_variance_plume(self,
                        _path,
                        logpath,
                        env,
                        extx=(0, 600),
                        exty=(-360, 360),
                        save_path=None):

        vids = [i for i in glob.glob(os.path.join(_path, env, '*.h5'))]
        logs = [j for j in glob.glob(os.path.join(_path, logpath, '*.csv'))]

        varls = []

        for vi, v in enumerate(vids):
            with h5py.File(v, 'r') as h5:
                mat = h5['frames'][:]
                mean = np.mean(mat, axis=0)
                # print(mat.shape)
                idx = np.linspace(0, mat.shape[0] - 1,
                                  int(mat.shape[0] / 30)).astype('int')
                N = int(mat.shape[0] / 30)
                # for k in range(int(mat.shape[0] / 100)):
                for k in tqdm(idx, ncols=0, desc=f'{vi+1}/{len(vids)}'):
                    varls.append(mat[k] - mean)

        var = np.sum(np.array(varls)**2, axis=0) / N
        std = np.sqrt(var)

        sns.set_style('white')
        sns.set_context(self.context)
        ext = extx + exty
        fig, ax = plt.subplots()
        img = ax.imshow(std, extent=ext, cmap='viridis')

        # ax.add_artist(
        #     Circle((0, 0),
        #            50,
        #            color='r',
        #            fill=False,
        #            linestyle='--',
        #            linewidth=0.5,
        #            zorder=1))

        for l in np.random.choice(logs, 20):
            df = pd.read_csv(l, usecols=[2, 3])
            df.columns = ['x', 'y']
            # ax.plot(df.x,
            #         df.y,
            #         linewidth=.5,
            #         alpha=0.3,
            #         color='white',
            #         zorder=3)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.savefig(save_path + '.svg')

        plt.show()

    def plot_plume_snapshots(self,
                             _path,
                             env,
                             extx=(0, 600),
                             exty=(-360, 360),
                             save_path=None):

        vids = [i for i in glob.glob(os.path.join(_path, env, '*.h5'))]
        vids = np.random.choice(vids, 2)

        mats = []

        for v in vids:
            with h5py.File(v, 'r') as h5:
                mat = h5['frames'][66]
                mats.append(mat)

        sns.set_style('white')
        sns.set_context(self.context)
        ext = extx + exty
        fig, ax = plt.subplots()
        img = ax.imshow(mats[0], extent=ext, cmap='magma', vmin=0, vmax=255)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img, cax=cax)

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.savefig(save_path + '.svg')

        plt.show()

    def plot_actions(self,
                    df,
                    n_actions,
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

        sns.set_style(self.style)
        sns.set_context(self.context)

        fig, ax = plt.subplots(figsize=self.figsize)
        _n_actions = len(df[hue].unique())
        _palette = sns.color_palette("husl", n_actions)
        _markers = ['s', '>', '^', 'v']

        if _n_actions < n_actions:

            _palette = _palette[(n_actions - _n_actions):]
            _markers = _markers[(n_actions - _n_actions):]

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

    

    def plot_reward_function(self, df, xlabel, lg, save_path=None, _logx=False):
        """Plot reward function for each state

        Args:
            df (pandas.DataFrame): Data from csv
            yticks (list of str): Labels for y-ticks
        """

        sns.set_style(self.style)
        sns.set_context(self.context)

        fig, ax = plt.subplots(figsize=self.figsize)
        # fig, axs = plt.subplots(4, 1, figsize=self.figsize, sharex=True)

        # for i, ax in enumerate(axs.flat):
        #     sns.lineplot(x='tblank',
        #                  y='Reward',
        #                 #  hue='Antennae',
        #                 #  style='Antennae',
        #                  data=df[df.Antennae == i],
        #                  ax=ax)

        ax = sns.lineplot(
            x='tblank',
            y='Reward',
            hue='Antennae',
            style='Antennae',
            markers=True,
            style_order=[3, 2, 1, 0],
            palette='Dark2',
            data=df)

        # ax.invert_yaxis()

        ax.set_xlabel(xlabel)
        if _logx:
            ax.set_xscale('log')
        ax.set_ylabel('Reward')
        l = ax.legend_
        for i, j in enumerate(lg):

            l.texts[i].set_text(j)
            # l.texts[1].set_text(lg[1])
        # cbar = fig.colorbar()
        # cbar.set_label('Reward')

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.savefig(save_path + '.svg')
        plt.show()


    def plot_policy(self, df, xticks):

        sns.set_style(self.style)
        sns.set_context(self.context)

        fig, ax = plt.subplots(figsize=self.figsize)
        plt.show()

    def kdeplots(self, df, xvars, col, output=None):

        sns.set_style('ticks')
        sns.set_context('paper')
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
        # z = df[col].to_numpy()
        # df[col] = np.cumsum(np.diff(z, prepend=z[0])**2)
        # df = df[['Time', 'x', 'y', col]].copy()# .rolling(40).mean()
        # xvars = ['x', 'y']
        # df[col] = df[col].rolling(40).sum()

        # axs[0].plot(df['x'], df[col])
        # axs[1].plot(df['y'], df[col])
        # for i, g in df.groupby((df.Time.diff() < 0).cumsum()):
        #     axs[0].plot(g.x,
        #                 g.burstiness,
        #                 linewidth=.2,
        #                 alpha=0.5,
        #                 color='k',
        #                 zorder=3)
        #     axs[1].plot(g.y,
        #                 g.burstiness,
        #                 linewidth=.2,
        #                 alpha=0.5,
        #                 color='k',
        #                 zorder=3)
        # sns.lineplot(data=df, x='x', y=col, ax=axs[0])
        # sns.lineplot(data=df, x='y', y=col, ax=axs[1])
        for i, ax in enumerate(axs):
            sns.kdeplot(data=df, x=xvars[i], y=col, fill=True, ax=ax)
            # ax.axhline(y=-.07, linewidth=1, linestyle='--', c='r')
            # ax.set_ylim(-.12, 0)

        # axs[1].set_xlim(-100, 100)
        sns.despine(fig)
        fig.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    





# Plot all the moth trajectories
def plot_moth_trajectories(df, xlim=(0, 600), ylim=(-360, 360), src=(0, 0, 50), output=None):
    srcx, srcy, goal_radius = src
    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(3.4, 2.83))

    ax.add_artist(Circle((srcx, srcy), goal_radius, color='r', 
        fill=False, linestyle='--', linewidth=0.5, zorder=1))

    for i, g in df.groupby((df.Time.diff() < 0).cumsum()):
        ax.plot(g.x_mm, g.y_mm, linewidth=.2, alpha=0.5, color='k', zorder=3)
        ax.scatter(g.x_mm.iloc[-1], g.y_mm.iloc[-1], color='g', s=5)

    ax.scatter(srcx, srcy, marker='*', c='k')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(output + '_moth_trajectories', dpi=300)
    plt.show()


def plot_moth_xyjoint(df, xlim=(0, 600), ylim=(-360, 360), src=(0, 0, 50), output=None):
    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(3.4, 2.83))
    ax = sns.jointplot(x='x_mm', y='y_mm',
                        data=df, kind='kde',
                        xlim, ylim)
    fig.tight_layout()
    plt.savefig(output + '_moth_xyjoint', dpi=300)
    plt.show()


def plot_heatmap(df, x, y, z,
                xlabel=None,
                ylabel=None,
                title=None,
                output=None):

    df = df.iloc[np.arange(0, len(df), 3)]
    # df = df[['x', 'y', 'burstiness']].copy().pivot_table(
    #   z, y, x,
    #   aggfunc=np.mean,
    #   fill_value=0)
    # df_smooth = gaussian_filter(df, sigma=40)
    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(3.5, 2))
    # ax = sns.heatmap(df_smooth,
    #                  vmin=np.min(df_smooth),
    #                  vmax=np.max(df_smooth),
    #                  cmap="plasma",
    #                 #  xticklabels=np.arange(0, 600, 100),
    #                 #  yticklabels=np.arange(-300, 400, 100),
    #                  cbar=True)
    ax = sns.heatmap(df.pivot_table(z, y, x, aggfunc=np.mean,
                    fill_value=0),
                    center=0,
                    annot=False)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.savefig("heat_map", dpi=300)
    plt.show()


def plot_moth_heading(df):
    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax = sns.lineplot(x='tblank', y='heading', data=df)
    ax.set_ylabel('Delta heading (deg)')
    ax.set_xlabel('tblank')
    fig.tight_layout()
    sns.despine(fig)
    ax.set_xlim(0, 109 / 30)
    plt.savefig('tethered2020-moth-heading-v-tblank', dpi=300)
    plt.show()


def plot_hit_rate(df):
    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax = sns.lineplot(x='Time', y='hit_rate', data=df)
    ax.set_ylabel('Hit rate')
    ax.set_xlabel('Time')
    fig.tight_layout()
    sns.despine(fig)
    plt.savefig('moth-hitrate-v-time', dpi=300)
    plt.show()


def plot_kinematics(df, _x, ylim0, ylim1, logscale=False):
    sns.set_style('ticks')
    sns.set_context('paper')
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    sns.lineplot(data=df, x=_x, y='linear_vel', ax=axs[0])
    sns.lineplot(data=df, x=_x, y='angular_vel', ax=axs[1])

    if logscale:
        axs[0].set_xscale('log')

    axs[0].set_xlabel('Blank duration (s)')
    axs[0].set_ylabel('Linear vel. (mm/s)')
    axs[1].set_ylabel('Angular vel. (rad/s)')

    axs[0].set_ylim(ylim0)
    axs[1].set_ylim(ylim1)

    fig.tight_layout()
    sns.despine(fig)
    plt.savefig("kinematics", dpi=300)
    plt.show()


def plt_action_histograms(df, x, hue,
                              xlabel,
                              lg_title,
                              lg,
                              bins='auto',
                              binrange=None,
                              save_path=None):
    num2cat = {0: 'Stop', 1: 'Surge', 2: 'CCW', 3: 'CW'}
    df.loc[:, hue] = df[hue].map(num2cat)

    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax = plt.subplots(figsize=(3.4, 2))

    ax = sns.histplot(data=df,
                        x=x,
                        hue=hue,
                        element='step',
                    #   stat='frequency',
                        bins=bins,
                        binrange=binrange,
                        palette='Dark2',
                        fill=False)
    ax.set_xlabel(xlabel)
    l = ax.legend_
    l.set_title(lg_title)
    fig.tight_layout()
    plt.savefig("action_histograms", dpi=300)
    plt.show()



# example
# def plot_infomation(out_path, plot_type, demos, dataframe):
#     plotter = mdp_plots.MdpPlots('ticks', 'paper', (3.5, 2.6))
#     outpath = os.path.join(out_path, 'plots')

#     elif plot_type == 'actions':
#         plotter.plot_actions(
#             demos, 
#             4, 
#             'action', 
#             'Linear vel. (mm/s)',
#             'Angular vel. (rad/s)', 
#             '',
#             ['Stop', 'Surge', 'Turn CCW', 'Turn CW'],
#             outpath + '_mg')
#         plotter.plt_action_histograms(
#             demos.copy(),
#             'linear_vel',
#             'action',
#             'Linear vel (mm/s)',
#             'Action', ['Stop', 'Surge', 'Turn CCW', 'Turn CW'],
#             bins=32,
#             save_path=outpath + '_linv_v2')
#         plotter.plt_action_histograms(
#             demos.copy(),
#             'angular_vel',
#             'action',
#             'Angular vel (rad/s)',
#             'Action', ['Stop', 'Surge', 'Turn CCW', 'Turn CW'],
#             binrange=[-2*np.pi, 2*np.pi],
#             save_path=outpath + '_angv_v2')

    # elif plot_type == 'heatmap':
    #     plotter.plot_heatmap(demos, 'state_num_i', 'antennae',
    #                             'hits', 'Blank duration',
    #                             'Hit antenna', 'Cumulative hits',
    #                             outpath + '_mis_DS')

    #     plotter.plot_heatmap(demos, 'state_num_i', 'antennae', 'wind',
    #                             'Blank duration', 'Hit antenna',
    #                             'Wind direction', outpath + '_mis_DS')

    # elif plot_type == 'states':
    #     plotter.plot_states(demos.copy(), 'log_tblank', 'antennae',
    #                         r'$\log(1+\tau_b)$', 'Hit antennae',
    #                         ['None', 'Right', 'Left', 'Both'], save_path=outpath + '_logtb_v2')
    #     plotter.plot_states(demos.copy(), 'state_num_i', 'antennae',
    #                         r'Discretized $\tau_b$', 'Hit antennae',
    #                         ['None', 'Right', 'Left', 'Both'], save_path=outpath + '_disc_v2')