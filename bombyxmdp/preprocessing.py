import os
import glob
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd


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

    delta_theta = (theta[:-1] - theta[1:])
    # avoid abrupt changes from 2*pi to 0 or opposite
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
    df.index = (pd.date_range(0, periods=N, freq='{0:.2f}ms'.format(T0 * 1e3)))
    rescaled_df = df.resample(f'{T1 * 1e3:.2f}ms').pad()  #.mean()
    return rescaled_df


def merge_data(timeout=0):
    """
        Calculate linear and angular velocity from trajectories and timestamp
    """
    basepath = os.getcwd()
    input_dir = os.path.join(basepath, "data")
    csvs = list(glob.glob(os.path.join(input_dir, '*.csv')))
    dfs = []
    lengths = []
    success_runs = 0    # Number of successful trials

    for csvfile in tqdm(csvs, ncols=0, desc='Merging csv files'):
        # Read csv into a dataframe
        df = pd.read_csv(csvfile)
        df.columns = ['Time', 'x_mm', 'y_mm', 'theta_rad', 'antennae', 'wind']
 
        # Define time step duration as a convenience variable (0.0333s)
        tstep = df['Time'].iloc[1]

        # Calculate linear velocity
        df['linear_vel'] = get_linear_vel(df['x_mm'], df['y_mm'], tstep)

        # Calculate angular velocity
        theta = df['theta_rad'].to_numpy()
        df['theta_rad'] = theta
        df['angular_vel'] = get_angular_vel(theta, tstep)
        
        df['traveled_distance'] = (df['linear_vel']*tstep).cumsum()
        net_displacement = np.hypot(df['x_mm'].iloc[-1] - df['x_mm'].iloc[0],
            df['y_mm'].iloc[-1] - df['y_mm'].iloc[0])

        df['tortuosity'] = df['traveled_distance'] / net_displacement
        df['cdv'] = centerline_deviation(df['y_mm'], tstep)
        df['heading'] = np.cos(np.pi - df['theta_rad'])

        #######################################################
        hit_dict = {'N': 0, 'R': 1, 'L': 1, 'B': 1}
        df['hit_hz'] = df['antennae'].map(hit_dict)
        df['hit_hz']= df['hit_hz'].rolling(min_periods=1, window=30).sum()
        #######################################################

        # Remap values of odor and wind cues to numeric values
        antennae_dict = {'N': 0, 'R': 1, 'L': 2, 'B': 3}
        df['antennae'] = df['antennae'].map(antennae_dict)
        wind_dict = {'B': 0, 'R': 1, 'L': 2, 'F': 3}
        df['wind'] = df['wind'].map(wind_dict)

        # hits: count cumulative odor hits
        # whiff: binary value of odor detection
        whiff = (df['antennae'].to_numpy() > 0).astype('uint8')
        hits = (whiff[:-1] < whiff[1:]).cumsum()
        hits = np.insert(hits, 0, 0)
        df['whiff'] = whiff
        df['hits'] = hits.astype('int')

        # get number of hits per 1 second
        df.loc[:, 'hit_rate'] = df['whiff'].rolling(int(1 / tstep)).sum()

        # Get last hit (Both, Left, Right).  
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
        df['tblank'] = df['tblank'].mul(tstep)
        df['tblank'] = df.loc[:, 'tblank'].shift(1, fill_value=0)
        
        # Transform blank duration to log scale to reduce skewness
        df['log_tblank'] = np.log1p(df['tblank'].to_numpy())

        # Compute whiff duration
        df['twhiff'] = df.groupby(df.antennae.eq(0).cumsum()).cumcount(ascending=True)
        df['twhiff'] = df['twhiff'].mul(tstep)
        df['twhiff'] = df.loc[:, 'twhiff'].shift(1, fill_value=0)

        # Transform blank duration to log scale to reduce skewness
        twhiff = df['twhiff'].to_numpy()
        log_twhiff = np.log1p(twhiff)
        df['log_twhiff'] = log_twhiff

        # Future states
        df['tblank_k'] = df.loc[:, 'tblank'].shift(-1, fill_value=0)
        df['log_tblank_k'] = df.loc[:, 'log_tblank'].shift(-1, fill_value=0)
        df['antennae_k'] = df.loc[:, 'antennae'].shift(-1, fill_value=0)
        df['hits_k'] = df.loc[:, 'hits'].shift(-1, fill_value=0)
        df['wind_k'] = df.loc[:, 'wind'].shift(-1, fill_value=0)

        # Add column with experiment name
        df['experiment'] = os.path.basename(csvfile)

        if (np.sqrt(df['x_mm'].iloc[-1]**2 + df['y_mm'].iloc[-1]**2) <= 50):
            success_runs += 1

        # only use the data which is recorded within the timeout
        if df['Time'].iloc[-1] <= timeout:
            dfs.append(df)
            lengths.append(len(df.index))


    dfs = pd.concat(dfs, axis=0, ignore_index=True)

    print(f'Successful runs: {success_runs}/{len(csvs)}')
    return dfs, lengths


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