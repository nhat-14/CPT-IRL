# -*- coding: utf-8 -*-
"""
Main console script for the silkmoth simulator
"""

from parse_arg import parse_args
import sys
import os
import glob
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm
import config
from utils import fileIO
import simulator

def main(args):
    args = parse_args(args)
    main_config = config.BombyxsimConf(args.conf)
    # field_conf = config.FieldConf(main_config)

    env = main_config.env_type
    agt = main_config.agent_type
    controller = main_config.controller_type

    plumes = [
        i for i in glob.glob(os.path.join(args.input_dir, env, '*.h5'))
    ]
    tlim = args.tlim

    sim = simulator.Simulator(args.input_dir, main_config, tlim, 
                            env=env, agt=agt, controller=controller)

    out_dir = None
    if args.plt_traj:
        out_dir = '{}_{}_{}_{}runs_{}'.format(agt, env, controller[0], args.Nruns, fileIO.tstamp())
        out_dir = fileIO.make_dir(args.input_dir, out_dir)
    
    fig, ax = plt.subplots()
    ax.add_artist(Circle((0,0), 50, color='r', fill=False, linestyle='--', linewidth=0.5, zorder=1))
    ax.add_patch(patches.Rectangle((215,-100),20,200,edgecolor = 'k',facecolor = 'k',fill=True))
    
    for i in tqdm(range(args.Nruns), leave=False):
        plume = random.choice(plumes)

        experiment_id = os.path.basename(os.path.splitext(plume)[0])
        experiment_id = '{}_{}'.format(i, experiment_id)
        trajectory, log = sim.run(plume,
                                  hit_prob=args.hit_prob,
                                  draw_animation=args.animation,
                                  save_log=args.save_log)

        if out_dir is not None:
            if args.save_log:
                log.to_csv(os.path.join(out_dir,
                           '{}_log.csv'.format(experiment_id)),
                           index=False)
        ax.plot(trajectory.iloc[:,0], trajectory.iloc[:,1], linewidth=0.5, c='g')

    ax.scatter(0, 0, marker='*', c='k')
    ax.set_xlim(0,500)
    ax.set_ylim(-200,200)
    ax.set_aspect('equal')
    plt.savefig('trajectory.png', dpi=300)
    plt.show()

    p = pd.DataFrame(sim.performance)
    if out_dir is not None:
        fileIO.make_dir(args.input_dir, 'performance')
        p.to_csv(os.path.join(
            args.input_dir, 'performance',
            '{}_{}_{}runs_{}.csv'.format(env, controller[0], args.Nruns,
                                        fileIO.tstamp())),
                index=False)
    print('  Success rate: {:.4f} +- {:.4f}'.format(
        p['success_rate'].sum() / args.Nruns, p['success_rate'].std()))
    print('  Search time: {:.4f} +- {:.4f}'.format(
        p.loc[p.success_rate == 1, 'search_time'].mean(),
        p.loc[p.success_rate == 1, 'search_time'].std()))

if __name__ == "__main__":
    main(sys.argv[1:])