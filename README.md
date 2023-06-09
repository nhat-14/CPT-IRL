
# Bombyxsim

A simulator for olfactory searches mainly focused on the study of the male silkmoth *Bombyx mori* but aimed to be expandable to other type of agents. The simulator has three main components: **environment**, **agent**, and the agent's **controller**. The definition for different types of these components is located in their respective folders: `envs/`, `agents/`, and `controllers/`. The user can add new types of these components as needed.

# Usage

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
```

## Command parameters

### -i, --input-dir

Directory from where the odor plume data will be read. An example `--input-dir` would have a subfolder named `smokevid/` which contains .h5 files with the frames of smoke plume videos.


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
bombyxsim -i examples/ --conf examples/experiment1.json -n 1 -t 120
```

You can also run a single simulation and visualize that the movement of the agent makes sense by showing the animation. To do so, use the following command:

```console
bombyxsim -i examples/ --conf examples/experiment1.json -n 1 -t 120 --animation
```

The following command would save the results of the experiments into csv files:

```console
bombyxsim -i examples/ --conf examples/experiment1.json -n 1 -t 120 -v --save-csv
```






rlalgorithms:

```console
python app.py -i examples/ -a silkmoth --algorithm QL -e smokevid -c IRL --runs 1 -T 15
```





# Bombyx IRL

Extract reward and action-value functions with:
- Maximum Entropy Inverse Reinforcement Learning [(Ziebart et al. 2008)](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf?source=post_page---------------------------) and Maximum Entropy Deep Inverse Reinforcement Learning [(Wulfmeier et al. 2017)](https://journals.sagepub.com/doi/abs/10.1177/0278364917722396).

# Usage

## MaxEnt IRL

- Store the states and actions of the behavior you want to analyze in a folder, for example `/data`. Some example trajectories are provided [here](https://drive.google.com/file/d/1ZkKS-h6VS1Pr6MhKpP9av_yKt6z6PxyD/view?usp=sharing).
- Such states and actions must be in csv files with column names `state` and `action` like:


|state|action|
|-----|------|
|0		|1     |

- Set the values of discount, epochs, learning rate, trajectory length by editing the file `config.json`

```console
usage: maxent_bombyx [-h] [--version] [--save-csv] [--plots] --data-dir
                     DATA_DIR [-v] [-vv]

Estimates the reward function of a silkmoth based on MaxEnt IRL

**Example**: Calculate moth reward and action-value and save the heatmap for each of these as well as the values as csv:

**Linux**

```console
maxent_bombyx --data-dir /home/user/full/path/to/csvs --plots --save-csv -v
```

**Windows**

```console
maxent_bombyx --data-dir C:\Users\user\full\path\to\csvs --plots --save-csv -v

```

## Deep MaxEnt IRL

1.  Create a conda environment with `conda env create -f bombyx-irl-py35.yml`
2.  Follow steps (3) to (5) of section 2.1
3. Execute the command `deep_maxent_bombyx --help` to see the description of each of its parameters. The output should be as follows:

```console
optional arguments:
  -h, --help            show this help message and exit
  --version             show programs version number and exit
  --save-csv            Save extracted reward and action-values as csv
  --plots               Plot reward and action-value function
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path of the directory with the input data
  -E EPOCHS, --epochs EPOCHS
                        Training epochs for the gradient descent
  -T TRAJ_LEN, --trajectory-length TRAJ_LEN
                        Length in discrete time steps of the expert
                        demonstrations
  -G DISCOUNT, --discount DISCOUNT
                        Discount factor
  --l2-reg L2REG [L2REG ...]
                        Whether to use L2 regularization in the gradient
                        descent
  -v, --verbose         set loglevel to INFO
  -vv, --very-verbose   set loglevel to DEBUG
```

### -i, --input-dir

This folder contains the following files:

- Expert demonstrations, i.e. state-action trajectories (.csv files)
- Trajectory files need to have at least two columns named `state_i` and `action` (see MaxEnt example)
- State features: `_features.csv`
- State-transition probabilities: `trans_prob.py`
- Example trajectories, state features and transition probabilities can be found in the `example/` folder

### Configuration file

- In the file `config.json` you can tweak values such as the total number of possible states (`n_states`), number of sub-states (e.g. 16 values for blank duration and 4 for reacted antenna).
- An example config file would be as follows:

### Example command

Calculate the reward function and policy from the trajectories in the folder `example/` by 200 traning epochs, 100 steps trajectory length, and discount rate of 0.9

```console
deep_maxent_bombyx -i example/ -e 200 -t 100 -g 0.9 -l 0.01

python src\deep_maxent_bombyx.py -i data -e 200 -t 109 -g 0.9 -l 0.01 --save-csv --plots
```

The above command should give the following output:

```console
Shape of feature matrix: (64, 2)
NN structure: (4, 3); learning rate: 0.01
Mean reward: [0.04, 0.014, -0.078, 0.024]
```



# Bombyxsim

A simulator for olfactory searches mainly focused on the study of the male silkmoth *Bombyx mori* but aimed to be expandable to robots, etc.

## Instructions

### Set up
Run `python app.py` to run the simulations


### How to use

- Store frames of an odor plume as .h5 files in a folder
- You can configure various parameters on `config.py`:

### Usage example

For example, if you put your plume files in a folder called: /home/user/bombyxsim-example/

Then, run the command below to simulate 20 runs with a time limit of 150 seconds using a silkmoth agent under plumes obtained by Yanagawa's smoke videos using Kanzaki et al. Programmed Behavior (shortened to KPB).

```console
bombyxsim --i /home/user/bombyxsim-example/ -a silkmoth -e smokevid -c KPB --runs 20 -T 260 --plot-traj --save-log -v

python simulator\src\bombyxsim\app.py --i 
```