import os
import glob
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
import config

def get_linear_vel(x, y, dt):
    """Get linear velocity based on x, y position vectors
    Args:
        x (pandas.Series): vector of x positions
        y (pandas.Series): vector of y position
        dt(float): Time step duration
    Return:
        dr (pandas.Series) = linear velocity in units of x,y per second
    """

    x = x.to_numpy()
    y = y.to_numpy()

    # result is equivalent to Equivalent to sqrt(x1**2 + x2**2)
    dR = np.hypot((x[:-1] - x[1:]), (y[:-1] - y[1:]))
    # insert 0 at 0 index to have the same length with
    dR = np.insert(dR, 0, 0)
    dR /= dt
    dR = pd.Series(dR)
    return dR


def get_angular_vel(theta, dt):
    """Get angular velocity based on x, y position vectors
    Args:
        theta (pandas.Series): vector of heading angles (rad)
        dt (float): Time step duration
    Return:
        dTh (pandas.Series) = angular velocity in units of rad per second
    """
    theta = theta.to_numpy()
    delta_theta = (theta[:-1] - theta[1:])
    # # avoid abrupt changes from 2*pi to 0 or opposite
    delta_theta = np.where(delta_theta > 5.0, delta_theta - 2*np.pi, delta_theta)
    delta_theta = np.where(delta_theta < -5.0, delta_theta + 2*np.pi, delta_theta)
    delta_theta = np.insert(delta_theta, 0, 0)
    return delta_theta / dt


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
    return list(glob.glob(os.path.join(input_dir, '*.csv')))


def is_source_found(x_pos_log, y_pos_log):
    """
    Check if the agent reach the source or not
    The source positon is set at coordinate (0,0) 
    """
    last_x = x_pos_log.iloc[-1]
    last_y = y_pos_log.iloc[-1]
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


def merge_data(timeout=0):
    """
    Calculate linear and angular velocity from trajectories and timestamp
    """
    dataframe_list = [] # List of data of each trials
    traj_len_list = []  # List of the length of each trajectories
    no_success_runs = 0 # Number of successful trials
    csv_list = get_csv_files()

    for csvfile in tqdm(csv_list, ncols=0, desc='Merging csv files'):
        # Read csv into a dataframe
        df = pd.read_csv(csvfile)
        df.columns = ['Time', 'x_mm', 'y_mm', 'theta_rad', 'antennae', 'wind']

        # only use the trial which data is recorded within the timeout
        if df['Time'].iloc[-1] > timeout:
            continue

        # Define time step duration as a convenience variable (0.0333s)
        time_step = df['Time'].iloc[1]

        # calculating simple moving average
        # using .rolling(window).mean() ,
        # with window size = 15. 0.5s
        # Calculate linear velocity and angular velocity
        df['linear_vel'] = get_linear_vel(df['x_mm'], df['y_mm'], time_step)
        df['angular_vel'] = get_angular_vel(df['theta_rad'], time_step)
        
        # moving average 
        df['linear_vel'] = df['linear_vel'].rolling(450).mean()
        df['angular_vel'] = df['angular_vel'].rolling(450).mean()
        # removing all the NULL values using
        df.dropna(inplace=True)

        import matplotlib.pyplot as plt
  
       
        # create figure and axis objects with subplots()
        fig,ax = plt.subplots()
        # make a plot
        ax.plot(df.iloc[:]['Time'].tolist(),
                df.iloc[:]['linear_vel'].tolist(),
                color="red")
        # set x-axis label
        ax.set_xlabel("time (s)", fontsize = 14)
        # set y-axis label
        ax.set_ylabel("linear vel (mm/s)",
                    color="red",
                    fontsize=14)

        # twin object for two different y-axis on the sample plot
        ax2=ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(df.iloc[:]['Time'].tolist(), df.iloc[:]['angular_vel'].tolist(),color="blue")
        ax2.set_ylabel("angular vel (rad/s)",color="blue",fontsize=14)
        plt.show()
        # save the plot as a file
        # fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
        #             format='jpeg',
        #             dpi=100,
        #             bbox_inches='tight')






        # Calculate some characteristic of trajectory paterns
        df['traveled_distance'] = (df['linear_vel']*time_step).cumsum()
        df['tortuosity'] = cal_tortuosity(df['x_mm'], df['y_mm'], df['traveled_distance'])
        df['cdv'] = centerline_deviation(df['y_mm'], time_step)
        df['heading'] = np.cos(np.pi - df['theta_rad'].to_numpy())

        # Remap values of odor and wind cues to numeric values
        antennae_dict = {'N': 0, 'R': 1, 'L': 2, 'B': 3}
        df['antennae'] = df['antennae'].map(antennae_dict)
        wind_dict = {'B': 0, 'R': 1, 'L': 2, 'F': 3}
        df['wind'] = df['wind'].map(wind_dict)

        # hits_count: count cumulative odor hits
        # whiff: binary value of odor detection
        df['whiff'] = (df['antennae'].to_numpy() > 0).astype('uint8')
        df['hits_count'] = count_hits(df['whiff'])
        
        # get number of hits per 1 second
        df.loc[:, 'hit_rate'] = df['whiff'].rolling(int(1 / time_step), min_periods=1).sum()

        # Get last hit (Both, Left, Right)
        df.loc[:, 'lasthit'] = df[df.antennae > 0].antennae
        df['lasthit'].fillna(method='ffill', inplace=True)
        # ffill: propagate last hit forward when no recent hits
        df['lasthit'] = df.loc[:, 'lasthit'].shift(1, fill_value=0)
        df['lasthit'].fillna(0, inplace=True)
        df['lasthit'] = df.lasthit.astype('uint8')

        # Calculate blank duration
        # no hit during tblank so no change in hit_cum
        hit_cum = df.antennae.gt(0).cumsum()
        df['tblank'] = df.groupby(hit_cum).cumcount(ascending=True)
        df['tblank'] = df['tblank'].mul(time_step)
        df['tblank'] = df.loc[:, 'tblank'].shift(1, fill_value=0)
        
        # Transform blank duration to log scale to reduce skewness
        df['log_tblank'] = np.log1p(df['tblank'].to_numpy())

        # Compute whiff duration
        df['twhiff'] = df.groupby(df.antennae.eq(0).cumsum()).cumcount(ascending=True)
        df['twhiff'] = df['twhiff'].mul(time_step)
        df['twhiff'] = df.loc[:, 'twhiff'].shift(1, fill_value=0)

        # Transform blank duration to log scale to reduce skewness
        twhiff = df['twhiff'].to_numpy()
        log_twhiff = np.log1p(twhiff)
        df['log_twhiff'] = log_twhiff

        # Future states
        df['tblank_k'] = df.loc[:, 'tblank'].shift(-1, fill_value=0)
        df['log_tblank_k'] = df.loc[:, 'log_tblank'].shift(-1, fill_value=0)
        df['antennae_k'] = df.loc[:, 'antennae'].shift(-1, fill_value=0)
        df['hits_count_k'] = df.loc[:, 'hits_count'].shift(-1, fill_value=0)
        df['wind_k'] = df.loc[:, 'wind'].shift(-1, fill_value=0)

        # Add column with experiment name as the csv file name
        df['experiment'] = os.path.basename(csvfile)

        dataframe_list.append(df)

        if is_source_found(df['x_mm'], df['y_mm']):
            no_success_runs += 1

    dataframe_list = pd.concat(dataframe_list, axis=0, ignore_index=True)
    print(f'Successful runs: {no_success_runs}/{len(csv_list)}')
    return dataframe_list


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