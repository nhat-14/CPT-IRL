# Bombyx IRL

Extract reward and action-value functions with:
- Maximum Entropy Inverse Reinforcement Learning [(Ziebart et al. 2008)](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf?source=post_page---------------------------) and Maximum Entropy Deep Inverse Reinforcement Learning [(Wulfmeier et al. 2017)](https://journals.sagepub.com/doi/abs/10.1177/0278364917722396).

# Usage

## MaxEnt IRL

- Store the states and actions of the behavior you want to analyze in a folder, for example `/path/to/csvs`. Some example trajectories are provided [here](https://drive.google.com/file/d/1ZkKS-h6VS1Pr6MhKpP9av_yKt6z6PxyD/view?usp=sharing).
- Such states and actions must be in csv files with column names `state` and `action` like:


|state|action|
|-----|------|
|0		|1     |

- Set the values of discount, epochs, learning rate, trajectory length by editing the file `config.json`
- Type `maxent_bombyx --help` in the terminal to see how to use the script. It should output this: 

```console
usage: maxent_bombyx [-h] [--version] [--save-csv] [--plots] --data-dir
                     DATA_DIR [-v] [-vv]

Estimates the reward function of a silkmoth based on MaxEnt IRL

optional arguments:
  -h, --help           show this help message and exit
  --version            show programs version number and exit
  --save-csv           Save extracted reward and action-values as csv
  --plots              Plot reward and action-value function
  --data-dir DATA_DIR  Path of the directory with the input data
  -v, --verbose        set loglevel to INFO
  -vv, --very-verbose  set loglevel to DEBUG

```

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

```json
{
    "n_states": 64,
    "n_sub_states": [16, 4],
    "df_cols": ["state_i", "action"],
    "n_actions": 4,
    "action_labels": ["stop", "surge", "turn_ccw", "turn_cw"], 
    "draw_subgrid": false,
    "subgrid_ticks": [4, 8, 12],
    "state_names": ["tblank", "antennae"],
    "state_labels": {
        "hits": "Cumulative hits",
        "tblank": "Blank duration",
        "antennae": "Antennae input",
        "wind": "Wind input direction"
    },
    "substate_ticks": {
        "antennae": ["None", "Right", "Left", "Both"], 
        "wind": ["Left", "Front", "Right", "Back"]
    }
}
```

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