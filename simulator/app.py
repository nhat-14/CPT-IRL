"""
Main console script for the silkmoth simulator
"""

import os
import glob
import random
import pandas as pd
from tqdm import tqdm
from utils import fileIO
import simulator

if __name__ == "__main__":
    tlim = 250              # Simulation time limit in seconds
    env = "smokevid"        #Type of environment
    agt = "silkmoth"        #Type of agent
    controller = ['KPB']    #Type of controller: [KPB, IRL (specify policy file)]
    Nruns = 50               #Number of simulation runs
    animation = False       #Draw animation
    # Path of the directory with odor plume data
    input_dir = "bombyxsim-template"
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
        out_dir = f'{agt}_{env}_{controller[0]}_{Nruns}runs_{fileIO.tstamp()}'
        tmp_dir = fileIO.make_dir(input_dir, out_dir)

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

    print('Performance metrics:')
    p = pd.DataFrame(sim.performance)
    print('Success rate: {:.4f} +- {:.4f}'.format(
        p['success_rate'].sum() / Nruns, p['success_rate'].std()))
    print('Search time: {:.4f} +- {:.4f}'.format(
        p.loc[p.success_rate == 1, 'search_time'].mean(),
        p.loc[p.success_rate == 1, 'search_time'].std()))
    print('End of script')