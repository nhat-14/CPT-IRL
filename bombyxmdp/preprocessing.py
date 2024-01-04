import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import config as cfg
import utilities as util   



def get_df_from_csv(file_name) -> pd.DataFrame:
    """
    Return a full formated panda data frame from csv
    """
    data = pd.read_csv(file_name) 
    data.columns = ['Time', 'x_mm', 'y_mm', 'theta_rad', 'antennae', 'wind']
    data['experiment'] = os.path.basename(file_name)
    return data


def is_timeout(data) -> bool:
    """
    Check if an experiment has the run time larger than time limit
    """
    last_time = data['Time'].iloc[-1]
    return last_time > cfg.exp_timeout


def is_source_found(data) -> bool:
    """
    Check if the agent reach the source or not
    The source positon is set at coordinate (0,0) 
    """
    last_x = data['x_mm'].iloc[-1]
    last_y = data['y_mm'].iloc[-1]
    return np.sqrt(last_x**2 + last_y**2) <= cfg.goal_radii


def cal_travelled_distance(data):
    data['traveled_distance'] = (data['linear_vel'] * cfg.time_step).cumsum()


def cal_moth_heading(data):
    data['heading'] = np.cos(np.pi - data['theta_rad'].to_numpy())


def get_linear_vel(data):
    """Get linear velocity based on x, y position vectors
    change during the time step
    """
    x = data['x_mm'].to_numpy()
    y = data['y_mm'].to_numpy()
    dR = np.hypot((x[1:] - x[:-1]), (y[1:] - y[:-1]))
    # insert 0 at 0 index to have the same length with
    dR = np.insert(dR, 0, 0)
    data['linear_vel'] = dR/cfg.time_step


def get_angular_vel(data):
    """Get angular velocity based on vector of heading angles (rad)
    change during time step duration
    """
    theta = data['theta_rad'].to_numpy()
    d_theta = (theta[1:] - theta[:-1])
    d_theta = util.wrapping_angle_from_0to2pi(d_theta)
    # insert 0 at 0 index to have the same length with
    d_theta = np.insert(d_theta, 0, 0)
    data['angular_vel'] = d_theta / cfg.time_step


def cal_centerline_deviation(data):
    """
    Calculate the deviation of the moth movement along center line (y=0)
    """
    y_src = 0.0
    y = data['y_mm'].to_numpy()
    dy = (y - y_src) * cfg.time_step
    c = np.cumsum(dy)
    data['cdv'] = pd.Series(c)


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


def cal_tortuosity(data):
    """
    Calculate tortuosity: the ratio between the traveled distance
    over the moving distance as the crow flies of agent in one trial 
    """
    first_x = data['x_mm'].iloc[0]
    first_y = data['y_mm'].iloc[0]
    last_x = data['x_mm'].iloc[-1]
    last_y = data['y_mm'].iloc[-1]
    crow_flies_dist = np.hypot(last_x - first_x, last_y - first_y)
    data['tortuosity'] = data['traveled_distance']/crow_flies_dist


def count_hits(whiff):
    whiff = whiff.to_numpy()
    count = (whiff[:-1] < whiff[1:]).cumsum()
    return np.insert(count, 0, 0).astype('int')


def moving_averaged_velocity(data, window_size):
    data['linear_vel'] = data['linear_vel'].rolling(window_size).mean()
    data['angular_vel'] = data['angular_vel'].rolling(window_size).mean()
    data.dropna(inplace=True) # removing all the NULL values 


def fill_future_states(data):
    """
    Shift one time step to show what is the next state given current state
    """
    for state in ["tblank", "log_tblank", "antennae", "hits_count", "wind", "region_x", "wind_B", "wind_F", "wind_L", "wind_R"]:
        data[f'{state}_k'] = data.loc[:, state].shift(-1, fill_value=0)


def cal_last_hit(data):
    """
    Get last hit type (Both, Left, Right)
    """
    # if agent got hit, last is the current hit
    data.loc[:, 'lasthit'] = data[data.antennae > 0].antennae
    data['lasthit'] = data.loc[:, 'lasthit'].shift(1, fill_value=0)
    # ffill: propagate last hit forward when no recent hits
    data['lasthit'] = data['lasthit'].ffill()
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


def numerize_antennae(data):
    antennae_dict = {'N': 0, 'R': 1, 'L': 2, 'B': 3}
    data['antennae'] = data['antennae'].map(antennae_dict)

    # one_hot = pd.get_dummies(data['antennae'])
    # data = data.join(one_hot)
    # antennae_dict = ('N', 'R', 'L', 'B')
    # data.columns = [f'antennae_{x}' if x in antennae_dict else x for x in data.columns]
    # return data


def numerize_wind(data):
    # wind_dict = {'B': 0, 'R': 1, 'L': 2, 'F': 3}
    # data['wind'] = data['wind'].map(wind_dict)

    one_hot = pd.get_dummies(data['wind'])
    # # Drop column as it is now encoded
    # data = data.drop('wind',axis = 1)
    # Join the encoded df
    data = data.join(one_hot)
    wind_dict = ('B', 'R', 'L', 'F')
    data.columns = [f'wind_{x}' if x in wind_dict else x for x in data.columns]
    return data

def cal_whiff_duration(data):
    dt = cfg.time_step
    data['twhiff'] = data.groupby(data.antennae.eq(0).cumsum()).cumcount(ascending=True)
    data['twhiff'] = data['twhiff'].mul(dt)
    data['twhiff'] = data.loc[:, 'twhiff'].shift(1, fill_value=0)
    
    # Use log scale to reduce skewness
    twhiff = data['twhiff'].to_numpy()
    log_twhiff = np.log1p(twhiff)
    data['log_twhiff'] = log_twhiff


def cal_time_blank(data):
    # Calculate blank duration. no hit during tblank => no change in hit_cum
    dt = cfg.time_step
    hit_cum = data.antennae.gt(0).cumsum()
    data['tblank'] = data.groupby(hit_cum).cumcount(ascending=True)
    data['tblank'] = data['tblank'].mul(dt)
    data['tblank'] = data.loc[:, 'tblank'].shift(1, fill_value=0)
    
    # Use log scale to reduce skewness
    data['log_tblank'] = np.log1p(data['tblank'].to_numpy())


def cal_hit_related_features(data):
    # whiff: binary value of odor detection
    data['whiff'] = (data.antennae.to_numpy() > 0).astype('uint8')

    # hits_count: count cumulative odor hits
    data['hits_count'] = count_hits(data['whiff'])
    
    # get number of hits per 1 second
    dt = cfg.time_step
    data.loc[:, 'hit_rate'] = data['whiff'].rolling(int(1 / dt), min_periods=1).sum()


def cal_regions(data):
    """
    Determine the region of agent staying in obstacle region
    For rectangle obstacle => 2 regions (1 and 0)
    """
    devide_line_x = cfg.lower_corner[0] + cfg.width/2 
    upper_line_y = cfg.lower_corner[1] + cfg.length
    lower_line_y = cfg.lower_corner[1]
    data['region_x'] = 0
    data['region_y'] = 0
    data.loc[data['x_mm'] > devide_line_x, 'region_x'] = 1
    data.loc[data['y_mm'] > upper_line_y, 'region_y'] = 1
    data.loc[data['y_mm'] < lower_line_y, 'region_y'] = 2


def cal_heading_to_obstacle(data):
    """
    Determine the direction of heading of agent toward
    or outward the obstacle.
    """
    devide_line_x = cfg.lower_corner[0] + cfg.width/2 
    data.loc[data['x_mm'] > devide_line_x, 'region_x'] = 1
    data['heading_obs'] = 0
    data.loc[(data['x_mm'] > devide_line_x) & (data['heading'] > 0), 'heading_obs'] = 1
    data.loc[(data['x_mm'] < devide_line_x) & (data['heading'] < 0), 'heading_obs'] = 1


def cal_dist2obstcle(data):
    """
    Calculacte the distance of the agent and the obstacle at anytime.
    For asymmetric obstacle (in both x and y), obstacle is devidend into
    multiple components with its own center. 
    """
    data['obstacle_distance'] = np.inf
    # devide the obstacle into 10 parts
    for i in range(9): 
        x_obs = cfg.lower_corner[0] + cfg.width/2  
        y_obs = cfg.lower_corner[1] + (i+0.5)*(cfg.length/10)
        x_dist = data.x_mm.to_numpy() - x_obs
        y_dist = data.y_mm.to_numpy() - y_obs
        dist = np.sqrt(np.power(x_dist, 2) + np.power(y_dist, 2))
        lastest_dist = data['obstacle_distance'].to_numpy()
        data['obstacle_distance'] = np.minimum(dist, lastest_dist)


def extract_features(data):
    # ===== free-environment features extractation ====
    get_linear_vel(data)
    get_angular_vel(data)
    moving_averaged_velocity(data, cfg.window_size)
    cal_travelled_distance(data)
    cal_tortuosity(data)
    cal_centerline_deviation(data)
    cal_moth_heading(data)
    numerize_antennae(data)
    data = numerize_wind(data)
    cal_hit_related_features(data)
    cal_last_hit(data)
    cal_time_blank(data)
    cal_whiff_duration(data)
    
    # ===== obstacle regions features extractation ====
    cal_regions(data)
    cal_dist2obstcle(data)
    cal_heading_to_obstacle(data)
    fill_future_states(data)
    return data


def merge_data() -> pd.DataFrame:
    """
    Process raw data to useful data such as features for IRL
    """
    merged_df = pd.DataFrame()
    n_success = 0

    csv_list = util.get_csv_files(cfg.INPUT_DIR, '*.csv')
    for csvfile in csv_list:
        df = get_df_from_csv(csvfile)
        # only use the moth search with runtime within timeout 
        if not is_timeout(df): 
            n_success += int(is_source_found(df))
            df = extract_features(df)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    print(f'Successful runs: {n_success}/{len(csv_list)}')
    return merged_df