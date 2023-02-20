
# Bombyxsim

A simulator for olfactory searches mainly focused on the study of the male silkmoth *Bombyx mori* but aimed to be expandable to other type of agents. The simulator has three main components: **environment**, **agent**, and the agent's **controller**. The definition for different types of these components is located in their respective folders: `envs/`, `agents/`, and `controllers/`. The user can add new types of these components as needed.

# Installation

1. Install [Anaconda3](https://docs.anaconda.com/anaconda/install/)
2. Open a terminal (For Windows use anaconda prompt)
3. Create a conda environment with `conda env create -f bombyxsim.yml`
4. Press "y" when prompted and the conda environment will be created
5. Activate the environment with `conda activate bombyxsim`
6. Run `python setup.py develop` to build the executables

# Usage

After setup, the command `bombyxsim` becomes usable. Typing `bombyxsim --help` in the terminal should output the following information: 

```console
usage: bombyxsim [-h] [--version] -i INPUT_DIR [--conf CONF] [-a AGT] [-e ENV]
                 [-c CONTROLLER [CONTROLLER ...]] --runs NRUNS [-T TLIM]
                 [--hit-prob HIT_PROB] [-A] [--plot-traj] [--save-log] [-v]
                 [-vv]

A simulator for olfactory searches

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path of the directory with odor plume data
  -c CONF, --conf CONF  Path to config file (json)
  -n NRUNS, --nruns NRUNS
                        Number of simulation runs
  -t TLIM, --tlim TLIM  Simulation time limit in seconds
  --hit-prob HIT_PROB   Hit probability
  -a, --animation       Draw animation
  --plot-traj           Plot trajectories
  --save-csv            Save log to csv
  -v, --verbose         set loglevel to INFO
  -vv, --very-verbose   set loglevel to DEBUG
```

# Windows
python src/bombyxsim/app.py -i C:\Users\takeshipyui\OneDrive\IRL\chr-at-titech-bombyxsim-4b80652aeacc\examples -c C:\Users\takeshipyui\OneDrive\IRL\chr-at-titech-bombyxsim-4b80652aeacc\examples\config.json -n 1 -t 200 --plot-traj

# Mac
python src/bombyxsim/app.py -i /Users/kondo/Downloads/chr-at-titech-bombyxsim-4b80652aeacc/examples -c /Users/kondo/Downloads/chr-at-titech-bombyxsim-4b80652aeacc/examples/config.json -n 1 -t 150

## Command parameters

### -i, --input-dir

Directory from where the odor plume data will be read. An example `--input-dir` would have a subfolder named `smokevid/` which contains .h5 files with the frames of smoke plume videos.

### -c, --conf

Configuration file (.json) that defines the parameters of each experiment. It is recommended to make a config file for each experimental condition. An example config. file is shown below:

```json
{
    "FPS": 30,
    "env_type": "smokevid",
    "field": {
        "srcpos": [0, 0],
        "xlim": [0, 600],
        "ylim": [-360, 360],
        "width": 512,
        "height": 640,
        "goalr": 50
    },
    "controller_type":["ITX"],
    "irl":{
        "num_states": ["tblank", "hits"]
    },
    "agent_type":"infotaxis",
    "init_pose": [500.0, 300.0, 180.0],
    "random_start": true,
    "init_pose_eps_type": "randint",
    "init_pose_eps": [
        10.0,
        10.0,
        30.0
    ],
    "wind_angle": 0.0,
    "infotaxis":{
        "wind_speed": 0.65,
        "diffusivity": 0.1,
        "emission_rate": 1,
        "particle_lifetime":30,
        "grid_shape": [30, 36],
        "agent_size": 0.020,
        "agent_speed": 0.040,
        "src_radius":0.005
    }
}
```

The description of each parameter is shown below:

|parameter	|description                                               |
|-----------|----------------------------------------------------------|
|FPS		|Simulation frames per second (i.e. control frequency)     |
|env_type	|Type of environment ("smokevid", or "filament")           |
|srcpos		|Odor source position     |
|xlim		|Limits of the x-axis     |
|ylim		|Limits of the y-axis     |
|width      |Number of columns of each frame in .h5 files         |
|height     |Number of rows of each frame in .h5 files            |
|goalr      |radius of the goal area                     |
|controller_type| ITX: infotaxis, HYB: hybrid, KPB: programmed-behavior, IRL: IRL|
|init_pose  |initial pose ($x_0$, $y_0$, $\theta_0$) of the agent     |
|random_start| Whether to randomize the initial pose of the agent, i.e. $x_0 + \epsilon$|
|init_pose_eps_type| Type of $\epsilon$; "randint": integer, "uniform": float  |
|init_pose_eps| Range of values for $\epsilon$  |
|wind_angle |Heading of the wind, 0.0 means wind flowing in the positive x-axis|

### Usage example

The command below would run the experiment `experiment1.json` for 30 runs, with a time limit of 120 seconds.

```console
bombyxsim -i examples/  -n 1 -t 120
```

You can also run a single simulation and visualize that the movement of the agent makes sense by showing the animation. To do so, use the following command:

```console
bombyxsim -i examples/ --conf examples/experiment1.json -n 1 -t 120 --animation
```

The following command would save the results of the experiments into csv files:

```console
bombyxsim -i examples/ --conf examples/experiment1.json -n 1 -t 120 -v --save-csv
```