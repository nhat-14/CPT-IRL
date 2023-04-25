
# Bombyxsim

A simulator for olfactory searches mainly focused on the study of the male silkmoth *Bombyx mori* but aimed to be expandable to robots, etc.


## Prerequisites

- [Anaconda3](https://docs.anaconda.com/anaconda/install/)

## Instructions

### Set up

1. Open a terminal (For Windows use anaconda prompt)
2. Create a conda environment with `conda env create -f bombyxsim.yml`
3. Press "y" when prompted and the conda environment will be created
4. Activate the environment with `conda activate bombyxsim`
5. Run `python setup.py develop` to build the executables


### How to use

- Store frames of an odor plume as .h5 files in a folder
- You can configure various parameters on `config.json` such as:

|parameter	|description|
|-----	|------|
|FPS		|Simulation frames per second (i.e. control frequency)     |
|srcpos		|Odor source position     |
|xspace		|limits of the environment in the x-axis     |
|yspace		|limits of the environment in the y-axis     |
|width      |width in pixels of the smoke videos         |
|height     |height in pixels of the smoke videos        |
|goalr      |radius of the goal area                     |
|init_pose  |initial pose [x, y, theta] of the agent     |
|wind_angle |heading of the wind, 0.0 means -->          |


- Type `bombyxsim --help` in the terminal to see how to use the script. It should output this: 

```console
usage: bombyxsim [-h] [--version] -i INPUT_DIR -a AGT -e ENV -c CONTROLLER
                 --runs NRUNS [-T TLIM] [-A] [--plot-traj] [--save-log] [-v]
                 [-vv]

A simulator for olfactory searches

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path of the directory with odor plume data
  -a AGT, --agent AGT   Type of agent: [silkmoth, ]
  -e ENV, --env ENV     Type of environment: [smokevid, ]
  -c CONTROLLER, --controller CONTROLLER
                        Type of controller: [KPB (Kanzaki et al. programmed
                        behavior), ]
  --runs NRUNS          Number of simulation runs
  -T TLIM, --tlim TLIM  Simulation time limit in seconds
  -A, --animation       Draw animation
  --plot-traj           Plot trajectories
  --save-log            Save log to csv
  -v, --verbose         set loglevel to INFO
  -vv, --very-verbose   set loglevel to DEBUG
```

### Usage example

For example, if you put your plume files in a folder called: /home/user/bombyxsim-example/

Then, run the command below to simulate 20 runs with a time limit of 150 seconds using a silkmoth agent under plumes obtained by Yanagawa's smoke videos using Kanzaki et al. Programmed Behavior (shortened to KPB).

```console
bombyxsim --i /home/user/bombyxsim-example/ -a silkmoth -e smokevid -c KPB --runs 20 -T 260 --plot-traj --save-log -v

python simulator\src\bombyxsim\app.py --i 
```