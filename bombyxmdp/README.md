# Bombyxmdp

Represents raw data from *Bombyx mori* experiments into MDP expert demonstrations that can be later plugged into an Inverse RL algorithm.

### How to use

Store trajectories obtained from Moth VR experiments as .csv files in a folder
Generate state-action trajectories from Shigaki's 2020 W+O tethered system logs

optional arguments:
  -p [PLOT], --plot [PLOT]
                        Plot reward and action-value function
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path of the directory with the input data
  --save-excel [SAVE_EXCEL]
                        Save descriptive stats to xlsx
  --save-csv [SAVE_CSV]
                        Save merged dataframe to csv
```

### Example

- Having a folder named data/ with a .csv files, run the following command:

```console
python app.py -i data/ --save-csv --plot
```

The above command will create the following items:
  - A new folder with .csv files containing [Time, state, action] trajectories of the "moth MDP"
  - A file named `trans_prob.npy` which is a states x actions x states 3D transition probability matrix
  - A plot of the moth trajectories