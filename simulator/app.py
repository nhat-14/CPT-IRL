"""
Main console script for the silkmoth simulator
"""

__author__      = "Duc-Nhat Luong"
__copyright__   = "Copyright 2022, The CPT-IRL Project"
__credits__     = ["Duc-Nhat Luong, Cesar Hernandez-Reyes"]
__license__     = "MIT"
__maintainer__  = "Duc-Nhat Luong"
__email__       = "nhat.luongduc@gmail.com"


from os.path import join, basename, splitext
import glob
import random
import pandas as pd
from tqdm import tqdm
import simulator
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from datetime import datetime
from  multiprocessing.pool import ThreadPool
from pathlib import Path
import config as cfg


def tstamp():
    """
    Get the current time in format mm/dd/_hh/mm/ss
    """
    return datetime.now().strftime('%m%d_%H%M%S')


def make_dir(basepath):
    """
    Create directory if it doesn't exist
    """
    out_dir = f'{cfg.agt}_{cfg.env}_{cfg.controller[0]}_{cfg.Nruns}runs_{tstamp()}'
    Path(join(basepath, out_dir)).mkdir(parents=True, exist_ok=True)
    return join(basepath, out_dir)


def print_simulation_result(performance):
    """
    print the simulation result on the screen
    """
    print('Performance metrics:')
    p = pd.DataFrame(performance)
    success_rate_mean = p['success_rate'].sum() / cfg.Nruns
    success_rate_std = p['success_rate'].std()
    search_time_mean = p.loc[p.success_rate == 1, 'search_time'].mean()
    search_time_std = p.loc[p.success_rate == 1, 'search_time'].std()
    print(f'Success rate: {success_rate_mean} +- {success_rate_std}')
    print(f'Search time: {search_time_mean} +- {search_time_std}')


def get_list_of_plumes():
    """
    return a list of h5 files (full path name) which describle the plumes
    """
    name_format = join(cfg.input_dir, cfg.env, '*.h5')
    return [i for i in glob.glob(name_format)]


def test(i):
    print(f"Simulation trial {i}")
    plume = random.choice(plumes)
    experiment_id = f'{i}_{basename(splitext(plume)[0])}'
    trajectory, log = sim.run(plume, draw_animation=cfg.animation, save_log=cfg.save_log)
    output_dir = make_dir(cfg.input_dir)
    log.to_csv(join(output_dir, f'{experiment_id}_log.csv'), index=False)

    # sim.plot_trajectory(trajectory, join(tmp_dir, experiment_id))
    ax.plot(trajectory.iloc[:,0], trajectory.iloc[:,1], linewidth=0.5, c='g')


if __name__ == "__main__":
    sim = simulator.Simulator(cfg.input_dir,
                              cfg.smoke_environment,
                              cfg.tlim,
                              env=cfg.env,
                              agt=cfg.agt,
                              controller=cfg.controller)
    
    fig, ax = plt.subplots()
    ax.add_artist(Circle((0,0), 50, color='r', 
                         fill=False, linestyle='--', 
                         linewidth=0.5, zorder=1))
    # ax.add_patch(Rectangle((1, 1), 20, 60))
    plumes = get_list_of_plumes()

    with tqdm(total=cfg.Nruns) as pbar: 
        with ThreadPool() as pool:
            for result in pool.map(test, range(cfg.Nruns)):
                pbar.update(1)

    # for i in tqdm(range(cfg.Nruns)):
    #     plume = random.choice(plumes)
    #     experiment_id = f'{i}_{basename(splitext(plume)[0])}'
    #     trajectory, log = sim.run(plume, draw_animation=cfg.animation, save_log=cfg.save_log)
    #     output_dir = make_dir(cfg.input_dir)
    #     log.to_csv(join(output_dir, f'{experiment_id}_log.csv'), index=False)

    #     # sim.plot_trajectory(trajectory, join(tmp_dir, experiment_id))
    #     ax.plot(trajectory.iloc[:,0], trajectory.iloc[:,1], linewidth=0.5, c='g')

    ax.scatter(0, 0, marker='*', c='k')
    ax.set_xlim(0,600)
    ax.set_ylim(-360,360)
    ax.set_aspect('equal')
    print_simulation_result(sim.performance)
    plt.show()


