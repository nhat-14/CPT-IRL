This project will systematically lead you through the process of utilizing trajectory data from male silkmoths as they locate female silkmoths, with the ultimate goal of designing an artificial agent for odor source localization. Data is obtained by recording the male silkmoths' behaviors in the mothVR system described in this [article](https://elifesciences.org/articles/72001).

# Usage 
## Data preprocessing and learning
The main program is in main.py where the raw data is processed firstly before passed for inverse reinforcement learning (IRL). The outcome of the IRL will be te reward function. 

To start data preprocessing and use IRL to get reward function, enter the following command: 

```
python3 main.py
```


# Bombyxsim

A simulator for olfactory searches mainly focused on the study of the male silkmoth *Bombyx mori* but aimed to be expandable to other type of agents. The simulator has three main components: **environment**, **agent**, and the agent's **controller**. 

The odor plume data will be read from .h5 files with the frames of smoke plume videos. Plumes obtained by Yanagawa's smoke videos using Kanzaki et al. Programmed Behavior (shortened to KPB). You can also run a single simulation and visualize that the movement of the agent makes sense by showing the animation. 

The description of each parameter is shown below:

|parameter	|description                                               |
|-----------|----------------------------------------------------------|
|FPS		  |Simulation frames per second (i.e. control frequency)     |
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