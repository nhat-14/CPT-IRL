from tqdm import tqdm
from scipy import stats
from os.path import isfile, join
from sklearn.preprocessing import PowerTransformer, RobustScaler, KBinsDiscretizer
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import sys
import logging
import numpy as np
import pandas as pd
import os
import glob
import json


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

    dR = np.hypot((x[:-1] - x[1:]), (y[:-1] - y[1:]))

    dR = np.insert(dR, 0, 0)
    dR /= dt

    dR = pd.Series(dR)

    return dR


def traveled_distance(x, y):

    x = x.to_numpy()
    y = y.to_numpy()

    dist = np.hypot((x[:-1] - x[1:]), (y[:-1] - y[1:]))
    dist = np.insert(dist, 0, 0)

    return pd.Series(dist)


def centerline_deviation(y, dt, y_src=0.):

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

    rescaled_df = df.resample('{0:.2f}ms'.format(T1 * 1e3)).pad()  #.mean()

    return rescaled_df


def merge_data(path_, ignore_idx=True, timeout=0):
    csvs = [i for i in glob.glob(os.path.join(path_, '*.csv'))]
    dfs = []
    lengths = []
    sr = 0

    for c in tqdm(csvs, ncols=0, desc='Merging csv files'):
        # Read csv into a dataframe
        df = pd.read_csv(c)
        df.columns = ['Time', 'x_mm', 'y_mm', 'theta_rad', 'antennae', 'wind']

        # Define time step duration as a convenience variable
        tstep = df['Time'].iloc[1]
        # tstep = df['Time'].iloc[0]
        # df['Time'] = df['Time'].sub(tstep)

        # Unwrap theta to avoid abrupt changes from 2*pi to 0
        theta = df['theta_rad'].to_numpy()
        df['theta_rad'] = theta

        # Calculate angular velocity
        # dTh = (theta[:-1] - theta[1:])
        # dTh = np.insert(dTh, 0, 0)
        dTh = np.gradient(theta)
        dTh = ((dTh + np.pi) % (2 * np.pi)) - np.pi
        df['angular_vel'] = dTh / tstep

        # Calculate linear velocity
        df['linear_vel'] = get_linear_vel(df['x_mm'], df['y_mm'], tstep)

        # Remove outliers in linear and angular velocity
        # z = np.abs(stats.zscore(df))
        z = np.abs(stats.zscore(df[['linear_vel', 'angular_vel']]))
        # df = df[(z <= 3).all(axis=1)]
        # df = df[(z <= 2).all(axis=1)]

        df['traveled_distance'] = traveled_distance(df['x_mm'], df['y_mm']).cumsum()
        net_displacement = np.hypot(df['x_mm'].iloc[-1] - df['x_mm'].iloc[0], df['y_mm'].iloc[-1] - df['y_mm'].iloc[0])

        df['tortuosity'] = df['traveled_distance'] / net_displacement
        df['cdv'] = centerline_deviation(df['y_mm'], tstep)
        # df['heading'] = np.cos(np.pi - df['theta_rad'])
        df['heading'] = np.cos(np.pi - df['theta_rad'])

        # Remap values of odor and wind cues to numeric values
        antennae_dict = {'N': 0, 'R': 1, 'L': 2, 'B': 3}
        # antennae_dict = {0: 2, 1: 1, 2: 3, 3: 0}
        df['antennae'] = df['antennae'].map(antennae_dict)

        wind_dict = {'B': 0, 'R': 1, 'L': 2, 'F': 3}
        df['wind'] = df['wind'].map(wind_dict)
        # df['wind'] = (df['wind'].sub(4.0)).astype('int')

        # df = resample_data(df, tstep, 780)

        # Count cumulative odor hits
        whiff = (df['antennae'].to_numpy() > 0).astype('uint8')
        hits = (whiff[:-1] < whiff[1:]).cumsum()
        hits = np.insert(hits, 0, 0)
        df['whiff'] = whiff
        df['hits'] = hits.astype('int')
        df.loc[:, 'hit_rate'] = df['whiff'].rolling(int(1 / tstep)).sum()

        # Get last hit
        df.loc[:, 'lasthit'] = df[df.antennae > 0].antennae
        df['lasthit'].fillna(method='ffill', inplace=True)
        df['lasthit'] = df.loc[:, 'lasthit'].shift(1, fill_value=0)
        df['lasthit'].fillna(0, inplace=True)
        df['lasthit'] = df.lasthit.astype('uint8')

        # Compute blank duration
        df['tblank'] = df.groupby(
            df.antennae.gt(0).cumsum()).cumcount(ascending=True)
        df['tblank'] = df['tblank'].mul(tstep)
        df['tblank'] = df.loc[:, 'tblank'].shift(1, fill_value=0)

        # Transform blank duration to log scale to reduce skewness
        tblank = df['tblank'].to_numpy()
        log_tblank = np.log1p(tblank)
        df['log_tblank'] = log_tblank

        # Compute whiff duration
        df['twhiff'] = df.groupby(
            df.antennae.eq(0).cumsum()).cumcount(ascending=True)
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
        experiment_id = os.path.basename(os.path.splitext(c)[0])
        df['experiment'] = experiment_id
        if (np.sqrt(df['x_mm'].iloc[-1]**2 + df['y_mm'].iloc[-1]**2) <= 50):
            sr += 1


        if (timeout > 0):
            if df['Time'].iloc[-1] <= timeout:
                dfs.append(df)
                # if df['Time'].iloc[-1] > timeout: dfs.append(df)
                lengths.append(len(df.index))

        else:
            dfs.append(df)
            # lengths.append(len(df.index))


    if ignore_idx:
        dfs = pd.concat(dfs, axis=0, ignore_index=True)

    else:
        dfs = pd.concat(dfs, axis=0, ignore_index=False)

    print('Successful runs: ({}/{})'.format(sr, len(csvs)))
    print('Average trajectory length: {}'.format(
        pd.Series(lengths).describe().T))
    # print('Successful run lengths GCD: {}'.format(np.gcd.reduce(lengths)))
    # print(np.sum(np.log(lengths)))
    return dfs, lengths


def yeo_johnson_transform(x):
    """Fit and transform data using a yeo-johnson transformation

    Args:
        x (pandas.Series): Data to fit
    
    Return:
        x_yj (numpy array): Transformed data
        l (numpy array): Lambdas from yeo-johnson
    """

    yj = PowerTransformer(method='yeo-johnson', standardize=True)
    yj.fit(x)

    x_yj = yj.transform(x)

    return yj, x_yj


def discretize(df, kbins, strat_kmeans=False):

    if strat_kmeans:
        enc = KBinsDiscretizer(n_bins=kbins,
                               encode='ordinal',
                               strategy='kmeans')

    else:
        enc = KBinsDiscretizer(n_bins=kbins, encode='ordinal')

    enc.fit(df)

    km_edges = enc.bin_edges_
    km_transformed = enc.transform(df)

    return km_transformed, km_edges