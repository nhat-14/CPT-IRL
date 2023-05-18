"""
This file auto select the optimal feature for inverse reinforcement
learning using the regression tree
"""
import pandas as pd
import numpy as np
from os.path import join
import config as cfg
from decision_tree_regression import DecisionTreeRegressor

def get_full_demos():
    """
    Get full concatenated demostration data of silkmoth from VR system
    which is pre-prossed in bombyxmdp
    """
    csv_path = join(cfg.input_dir, "fmdp_demos.csv")
    return pd.read_csv(csv_path)


def get_latest_reward():
    csv_path = join(cfg.input_dir, "freward.csv")
    return pd.read_csv(csv_path)


def get_feature_matrix():
    """
    Get feature dataframe from the features.csv generated 
    from data processing step in bombyxmdp
    """
    feature_path = get_file_path('features.csv')
    matrix = pd.read_csv(feature_path).values
    print(f'Shape of feature matrix: {matrix.shape}')
    return matrix


def initialize_feature_matrix():
    """
    initialize_feature_matrix with unit vector.
    Which is a diagonal vector with shape of n_states x n_states
    """
    matrix = np.eye(cfg.n_states)
    print(f'Shape of feature matrix: {matrix.shape}')
    return matrix


# def get_reward_function():
#     """
#     Return the lastest reward function generated from inverse reinforcement
#     learning (optimization step) for feature selection (fitting step)
#     """
#     csv_path = load_csv_files("fmdp_demos.csv")
#     return pd.read_csv(csv_path)


def assign_reward_along_trajectory(dataframe, lookup):
    """
    Having the reward function, we use it to assign the
    state reward in each step of the moth demostration
    """
    dataframe['reward'] = dataframe.state_i.map(lookup.set_index('state').reward)
    dataframe.to_csv("test.csv", index=False)
    return dataframe

def get_data_set_for_fitting(data):
    # return data[['antennae', 'wind', 'linear_vel', 'angular_vel', 'traveled_distance', "whiff", "hit_rate", "lasthit", "tblank", "reward"]]
    return data[['antennae',"tblank", 'angular_vel', "hit_rate" ,"reward"]]



if __name__ == "__main__":
    full_demos_data = pd.read_csv(join(cfg.input_dir, "fmdp_demos.csv"))
    full_demos_data["reward"] = np.nan
    state_reward_table = get_latest_reward()
    full_demos_data = assign_reward_along_trajectory(full_demos_data, state_reward_table)
    small_demos_data = get_data_set_for_fitting(full_demos_data)
    # print(small_demos_data)

    
    X = small_demos_data.iloc[:2000, :-1].values
    Y = small_demos_data.iloc[:2000, -1].values.reshape(-1,1)
    # data = pd.read_csv("airfoil_self_noise.csv")
    # X = data.iloc[:, :-1].values
    # Y = data.iloc[:, -1].values.reshape(-1,1)
    regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
    data_size = 2000
    regressor.fit(X,Y)
    regressor.print_tree()

    # X_test = small_demos_data.iloc[2000:2020, :-1].values
    # Y_test = small_demos_data.iloc[2000:2020, -1].values
    # Y_pred = regressor.predict(X_test) 

    # print(Y_pred)
    # from sklearn.metrics import mean_squared_error
    # print(np.sqrt(mean_squared_error(Y_test, Y_pred)))