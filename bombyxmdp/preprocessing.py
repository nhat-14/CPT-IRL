import os
import glob
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer

import config

def get_linear_vel(data):
    """Get linear velocity based on x, y position vectors
    change during the time step
    """
    dt = get_time_step(data)
    x = data['x_mm'].to_numpy()
    y = data['y_mm'].to_numpy()
    dR = np.hypot((x[1:] - x[:-1]), (y[1:] - y[:-1]))
    # insert 0 at 0 index to have the same length with
    dR = np.insert(dR, 0, 0)
    data['linear_vel'] = dR/dt


def wrapping_angle_from_0to2pi(angle):
    mask = angle > np.pi
    angle[mask] = angle[mask] - 2*np.pi
    mask = angle < -np.pi
    angle[mask] = angle[mask] + 2*np.pi


def get_angular_vel(data):
    """Get angular velocity based on vector of heading angles (rad)
    change during time step duration
    """
    dt = data.at[1, 'Time'] - data.at[0, 'Time']
    theta = data['theta_rad'].to_numpy()
    delta_theta = (theta[1:] - theta[:-1])
    wrapping_angle_from_0to2pi(delta_theta)
    # insert 0 at 0 index to have the same length with
    delta_theta = np.insert(delta_theta, 0, 0)
    data['angular_vel'] = delta_theta / dt


def centerline_deviation(y, dt, y_src=0.):
    """
    Calculate the deviation of the moth movement along center line (y=0)
    """
    y = y.to_numpy()
    dy = (y - y_src) * dt
    c = np.cumsum(dy)
    return pd.Series(c)


def resample_data(df, dt, tl):
    """Downsample data to the length specified
    Args:
        df (pandas.DataFrame): Pandas data frame to resample
        tl (int): Trajectory length of the resampled dataframe
        dt (int): Original sampling period in seconds
    Return: Rescaled dataframe
    """
    # Downsample data to trajectory length
    N = len(df)
    T0 = round(dt, 5)
    T1 = round((N / tl) * T0, 5)
    df.index = pd.date_range(0, periods=N, freq=f'{T0 * 1e3:.2f}ms')
    rescaled_df = df.resample(f'{T1 * 1e3:.2f}ms').pad()  #.mean()
    return rescaled_df


def get_csv_files():
    """
    Get csv files of trajectories obtained from Moth VR experiments
    in input folder specified in the config.py
    """
    input_dir = config.INPUT_DIR
    return list(glob.glob(join(input_dir, '*.csv')))


def check_is_source_found(data):
    """
    Check if the agent reach the source or not
    The source positon is set at coordinate (0,0) 
    """
    last_x = data['x_mm'].iloc[-1]
    last_y = data['y_mm'].iloc[-1]
    return np.sqrt(last_x**2 + last_y**2) <= config.GOAL_RADII


def cal_tortuosity(x_pos_log, y_pos_log, route_len):
    """
    Calculate tortuosity: the ratio between the traveled distance
    over the moving distance as the crow flies of agent in one trial 
    """
    first_x = x_pos_log.iloc[0]
    first_y = y_pos_log.iloc[0]
    last_x = x_pos_log.iloc[-1]
    last_y = y_pos_log.iloc[-1]
    crow_flies_dist = np.hypot(last_x - first_x, last_y - first_y)
    return route_len/crow_flies_dist


def count_hits(whiff):
    whiff = whiff.to_numpy()
    count = (whiff[:-1] < whiff[1:]).cumsum()
    return np.insert(count, 0, 0).astype('int')


def get_data_frame_from_csv(file_name):
    """
    Return a full formated panda data frame from csv
    """
    data = pd.read_csv(file_name)   # Read csv into a dataframe
    data.columns = ['Time', 'x_mm', 'y_mm', 'theta_rad', 'antennae', 'wind']
    return data


def check_is_timeout(data, time_limit):
    """
    Check if an experiment has the run time larger than time limit
    """
    return data['Time'].iloc[-1] > time_limit


def moving_averaged_velocity(data, window_size):
    data['linear_vel'] = data['linear_vel'].rolling(window_size).mean()
    data['angular_vel'] = data['angular_vel'].rolling(window_size).mean()
    data.dropna(inplace=True) # removing all the NULL values 


def fill_future_states(data):
    """
    Shift one time step to show what is the next state given current state
    """
    for state in ["tblank", "log_tblank", "antennae", "hits_count", "wind"]:
        data[f'{state}_k'] = data.loc[:, state].shift(-1, fill_value=0)


def set_last_hit(data):
    """
    Get last hit type (Both, Left, Right)
    """
    # if agent got hit, last is the current hit
    data.loc[:, 'lasthit'] = data[data.antennae > 0].antennae
    # ffill: propagate last hit forward when no recent hits
    data['lasthit'].fillna(method='ffill', inplace=True)
    data['lasthit'] = data.loc[:, 'lasthit'].shift(1, fill_value=0)
    data['lasthit'].fillna(0, inplace=True)
    data['lasthit'] = data.lasthit.astype('uint8')


def discretize(dataframe, kbins, strat_kmeans=False):
    """ 
    Bin continuous tblank data into intervals.
    """
    if strat_kmeans:
        enc = KBinsDiscretizer(n_bins=kbins, encode='ordinal', strategy='kmeans')
    else:
        enc = KBinsDiscretizer(n_bins=kbins, encode='ordinal')
    enc.fit(dataframe)
    km_edges = enc.bin_edges_
    km_transformed = enc.transform(dataframe)
    return km_transformed, km_edges


def extract_velocity_from_pose(data):
    get_linear_vel(data)
    get_angular_vel(data)
    moving_averaged_velocity(data, config.WINDOW_SIZE)


def numerize_antennae(data):
    antennae_dict = {'N': 0, 'R': 1, 'L': 2, 'B': 3}
    data['antennae'] = data['antennae'].map(antennae_dict)


def numerize_wind(data):
    wind_dict = {'B': 0, 'R': 1, 'L': 2, 'F': 3}
    data['wind'] = data['wind'].map(wind_dict)


def get_time_step(data):
    dt = data['Time'].iloc[1] - data['Time'].iloc[0]
    return dt


def cal_whiff_duration(data):
    dt = get_time_step(data)
    data['twhiff'] = data.groupby(data.antennae.eq(0).cumsum()).cumcount(ascending=True)
    data['twhiff'] = data['twhiff'].mul(dt)
    data['twhiff'] = data.loc[:, 'twhiff'].shift(1, fill_value=0)
    
    # Use log scale to reduce skewness
    twhiff = data['twhiff'].to_numpy()
    log_twhiff = np.log1p(twhiff)
    data['log_twhiff'] = log_twhiff


def set_time_blank(data):
    # Calculate blank duration. no hit during tblank => no change in hit_cum
    dt = get_time_step(data)
    hit_cum = data.antennae.gt(0).cumsum()
    data['tblank'] = data.groupby(hit_cum).cumcount(ascending=True)
    data['tblank'] = data['tblank'].mul(dt)
    data['tblank'] = data.loc[:, 'tblank'].shift(1, fill_value=0)
    
    # Use log scale to reduce skewness
    data['log_tblank'] = np.log1p(data['tblank'].to_numpy())


def extract_hit_related_features(data):
    # whiff: binary value of odor detection
    data['whiff'] = (data.antennae.to_numpy() > 0).astype('uint8')

    # hits_count: count cumulative odor hits
    data['hits_count'] = count_hits(data['whiff'])
    
    # get number of hits per 1 second
    dt = get_time_step(data)
    data.loc[:, 'hit_rate'] = data['whiff'].rolling(int(1 / dt), min_periods=1).sum()


def merge_data(timeout=0):
    """
    Calculate linear and angular velocity from trajectories and timestamp
    """
    dataframe_list = [] # List of data of each trials
    n_success_runs = 0

    csv_list = get_csv_files()
    for csvfile in tqdm(csv_list, ncols=0, desc='Merging csv files'):
        # only use the trial which data is recorded within the timeout
        df = get_data_frame_from_csv(csvfile)
        if check_is_timeout(df, timeout):
            continue
        
        if check_is_source_found(df):
            n_success_runs += 1

        # Define time step duration (0.0333s)
        time_step = df['Time'].iloc[1]
        extract_velocity_from_pose(df)
        
        # Calculate some characteristic of trajectory paterns
        df['traveled_distance'] = (df['linear_vel']*time_step).cumsum()
        df['tortuosity'] = cal_tortuosity(df['x_mm'], df['y_mm'], df['traveled_distance'])
        df['cdv'] = centerline_deviation(df['y_mm'], time_step)
        df['heading'] = np.cos(np.pi - df['theta_rad'].to_numpy())

        numerize_antennae(df)
        numerize_wind(df)
        extract_hit_related_features(df)
        set_last_hit(df)
        set_time_blank(df)
        cal_whiff_duration(df)
        fill_future_states(df)
        
        # Add column with experiment name as the csv file name
        df['experiment'] = os.path.basename(csvfile)
        dataframe_list.append(df)

    dataframe_list = pd.concat(dataframe_list, ignore_index=True)

    print(f'Successful runs: {n_success_runs}/{len(csv_list)}')
    return dataframe_list