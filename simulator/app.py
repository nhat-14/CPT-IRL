"""
Main console script for the silkmoth simulator
"""

import os
import glob
import random
import pandas as pd
from tqdm import tqdm
import simulator
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from pathlib import Path

def tstamp():
    return datetime.now().strftime('%m%d_%H%M%S')


def make_dir(basepath, filepath):
    # Make dir if it doesn't exist
    Path(os.path.join(basepath, filepath)).mkdir(parents=True, exist_ok=True)
    return os.path.join(basepath, filepath)


def print_simulation_result(performance):
    print('Performance metrics:')
    p = pd.DataFrame(performance)
    success_rate_mean = p['success_rate'].sum() / Nruns
    success_rate_std = p['success_rate'].std()
    search_time_mean = p.loc[p.success_rate == 1, 'search_time'].mean()
    search_time_std = p.loc[p.success_rate == 1, 'search_time'].std()
    print(f'Success rate: {success_rate_mean} +- {success_rate_std}')
    print(f'Search time: {search_time_mean} +- {search_time_std}')
    print('End of script')


if __name__ == "__main__":
    tlim = 250              # Simulation time limit in seconds
    env = "smokevid"        #Type of environment
    agt = "silkmoth"        #Type of agent
    controller = ['KPB']    #Type of controller: [KPB, IRL (specify policy file)]
    Nruns = 50              #Number of simulation runs
    animation = False       #Draw animation
    input_dir = "bombyxsim-template" # Path of the directory with odor plume data
    save_log = True
    plt_traj = True

    config_file = os.path.join(input_dir, env, 'config.json')
    plumes = [i for i in glob.glob(os.path.join(input_dir, env, '*.h5'))]

    sim = simulator.Simulator(input_dir,
                              config_file,
                              tlim,
                              env=env,
                              agt=agt,
                              controller=controller)
    
    if save_log or plt_traj:
        out_dir = f'{agt}_{env}_{controller[0]}_{Nruns}runs_{tstamp()}'
        tmp_dir = make_dir(input_dir, out_dir)

    fig, ax = plt.subplots()
    ax.add_artist(Circle((0,0), 50, color='r', fill=False, linestyle='--', linewidth=0.5, zorder=1))

    for i in tqdm(range(Nruns)):
        plume = random.choice(plumes)
        experiment_id = os.path.basename(os.path.splitext(plume)[0])
        experiment_id = f'{i}_{experiment_id}'

        if save_log:
            trajectory, log = sim.run(plume, draw_animation=animation, save_log=True)
            log.to_csv(os.path.join(tmp_dir, f'{experiment_id}_log.csv'), index=False)
        else:
            trajectory = sim.run(plume, draw_animation=animation)

        # sim.plot_trajectory(trajectory, os.path.join(tmp_dir, experiment_id))
        ax.plot(trajectory.iloc[:,0], trajectory.iloc[:,1], linewidth=0.5, c='g')

    ax.scatter(0, 0, marker='*', c='k')
    ax.set_xlim(0,600)
    ax.set_ylim(-360,360)
    ax.set_aspect('equal')
    plt.show()

    print_simulation_result(sim.performance)
    # print('Performance metrics:')
    # p = pd.DataFrame(sim.performance)
    # print(f'Success rate: {p['success_rate'].sum() / Nruns} +- {p['success_rate'].std()}')
    # print(f'Search time: {p.loc[p.success_rate == 1, 'search_time'].mean()} 
    #       +- {p.loc[p.success_rate == 1, 'search_time'].std()}')
    # print('End of script')