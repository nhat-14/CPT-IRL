# Bombyxmdp

Represents raw data from *Bombyx mori* experiments into MDP expert demonstrations that can be later plugged into an Inverse RL algorithm.

### How to use

- Store trajectories obtained from Moth VR experiments as .csv files in a folder
- You can configure various parameters on `config.json` such as:

|parameter	|description|
|-----------|-----------|
|srcxy		|Odor source position     |
|xlim		|limits of the environment in the x-axis     |
|ylim		|limits of the environment in the y-axis     |
|goal_radius|radius of the goal area                     |


- Type `rldemos-mothvr --help` in the terminal to see how to use the script. It should output this: 

```console
usage: rldemos-mothvr [-h] [--version] [-p [PLOT]] -i INPUT_DIR
                      [--save-excel [SAVE_EXCEL]]
                      [--save-trans-prob [SAVE_TRANS_PROB]]
                      [--save-csv [SAVE_CSV]] [-v] [-vv]

Generate state-action trajectories from Shigaki's 2020 W+O tethered system
logs

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -p [PLOT], --plot [PLOT]
                        Plot reward and action-value function
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path of the directory with the input data
  --save-excel [SAVE_EXCEL]
                        Save descriptive stats to xlsx
  --save-trans-prob [SAVE_TRANS_PROB]
                        Save transition probabilities to .npy
  --save-csv [SAVE_CSV]
                        Save merged dataframe to csv
  -v, --verbose         set loglevel to INFO
  -vv, --very-verbose   set loglevel to DEBUG
```

### Example

- Having a folder named Data/trajectories/ with a .csv files and a config.json file, run the following command:

```console
rldemos-mothvr -i Data/trajectories/ --save-csv --save-trans-prob --plot -v
```

- The above command will create the following items:
  - A new folder with .csv files containing [Time, state, action] trajectories of the "moth MDP"
  - A file named `trans_prob.npy` which is a states x actions x states 3D transition probability matrix
  - A plot of the moth trajectories