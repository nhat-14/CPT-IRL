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

def encode_actions_MG(df, columns):

    tb, lin_vel, ang_vel = tuple(columns)

    av_ccw, av_cw = (0.087, -0.087)

    surge = (df[tb].le(0.5) & df[lin_vel].gt(0)) | (df[tb].gt(0.2) & df[ang_vel].between(av_cw, av_ccw, inclusive=False))

    turn_ccw = ~(surge) & (df[ang_vel] > 0)
    turn_cw = ~(surge) & (df[ang_vel] < 0)

    for i, a in enumerate([surge, turn_cw, turn_ccw]):
        df.loc[a, 'action_mg'] = i + 1

    df['action_mg'].fillna(0, inplace=True)
    df['action_mg'] = df.action_mg.astype('uint8')

def encode_actions_KZ(df, columns):

    tb, _, ang_vel = tuple(columns)

    surge = df[tb].le(0.5)

    turn_ccw = ~(surge) & (df[ang_vel] >= 0)
    turn_cw = ~(surge) & (df[ang_vel] < 0)

    for i, a in enumerate([surge, turn_cw, turn_ccw]):
        df.loc[a, 'action_kz'] = i + 1

    df['action_kz'].fillna(0, inplace=True)
    df['action_kz'] = df.action_kz.astype('uint8')


def get_mismatching_expected_reward(df, reward_cols):

    df['match_pb'] = df.action_mg.eq(df.action_kz).astype('uint8')

    df['last_action_mg'] = df.loc[:, 'action_mg'].shift(1, fill_value=0)
    df['last_action_kz'] = df.loc[:, 'action_kz'].shift(1, fill_value=0)

    for i in sorted(df.action_kz.unique()):
        df.loc[(df.last_action_kz == i)
               & df.match_pb, 'EX_DS'] = df[reward_cols[i]]

    for j in sorted(df.action_mg.unique()):
        df.loc[(df.last_action_mg == j)
               & ~(df.match_pb), 'EX_DS'] = df[reward_cols[j]]


def merge_data(path_, ignore_idx=True):
    # Get paths of csv files
    csvs = [i for i in glob.glob(os.path.join(path_, '*.csv'))]

    # Store dataframes in a list
    dfs = []
    lengths = []

    for c in tqdm(csvs):
        # Read csv into a dataframe
        # df = pd.read_csv(c, header=None).T
        df = pd.read_csv(c, usecols=[0, 1, 2, 4, 5, 7, 15, 16, 17, 18])
        df.columns = ['Time', 'x_mm', 'y_mm', 'theta_rad', 'entropy', 'hit', 'r_stop', 'r_surge', 'r_turn_cw', 'r_turn_ccw']

        # Define time step duration as a convenience variable
        tstep = df['Time'].iloc[1]

        # Unwrap theta to avoid abrupt changes from 2*pi to 0
        theta = df['theta_rad'].to_numpy()
        df['theta_rad'] = theta

        # Calculate angular velocity
        # dTh = (theta[:-1] - theta[1:])
        # dTh = np.insert(dTh, 0, 0)
        dTh = np.gradient(theta)
        dTh = ((dTh + np.pi) % (2 * np.pi)) - np.pi
        df['angular_vel'] = dTh / tstep

        # Calculate real reward
        df.loc[:, 'DS'] = np.gradient(df['entropy'].to_numpy())

        # Calculate linear velocity
        df['linear_vel'] = get_linear_vel(df['x_mm'], df['y_mm'], tstep)

        # Remove outliers in linear and angular velocity
        z = np.abs(stats.zscore(df[['linear_vel', 'angular_vel']]))
        df = df[(z < 3).all(axis=1)]

        # Count cumulative odor hits
        hit = (df['hit'].to_numpy() > 0).astype('uint8')

        hitsum = (hit[:-1] < hit[1:]).cumsum()
        hitsum = np.insert(hitsum, 0, 0)
        df['hitsum'] = hitsum.astype('int')

        df.loc[:, 'hit_rate'] = df['hit'].rolling(int(1 / tstep)).sum()

        # Compute blank duration
        df['tblank'] = df.groupby(
            df.hit.gt(0).cumsum()).cumcount(ascending=True)
        df['tblank'] = df['tblank'].mul(tstep)
        df['tblank'] = df.loc[:, 'tblank'].shift(1, fill_value=0)

        # Transform blank duration to log scale to reduce skewness
        tblank = df['tblank'].to_numpy()
        log_tblank = np.log1p(tblank)
        df['log_tblank'] = log_tblank

        action_cols = ['tblank', 'linear_vel', 'angular_vel']
        reward_cols = ['r_stop', 'r_surge', 'r_turn_cw', 'r_turn_ccw']

        encode_actions_MG(df, action_cols)
        encode_actions_KZ(df, action_cols)
        get_mismatching_expected_reward(df, reward_cols)

        # Future states
        df['tblank_k'] = df.loc[:, 'tblank'].shift(-1, fill_value=0)
        df['log_tblank_k'] = df.loc[:, 'log_tblank'].shift(-1, fill_value=0)
        df['DS_k'] = df.loc[:, 'DS'].shift(-1, fill_value=0)
        df['hits_k'] = df.loc[:, 'hitsum'].shift(-1, fill_value=0)
        # df['wind_k'] = df.loc[:, 'wind'].shift(-1, fill_value=0)

        df.loc[:, 'RMSE'] = pd.Series(
            (df.DS_k - df.EX_DS)**2).rolling(int(1 / tstep)).mean()
        # df.loc[:, 'RMSE'] = np.sqrt(
        # np.cumsum((df.DS_k - df.EX_DS)**2) / df.index)
        df['RMSE'] = np.sqrt(df.RMSE)
        df = df.replace([np.inf, -np.inf], 0)
        # df['RMSE'] = df['RMSE'].rolling(int(1 / tstep)).mean()

        # Add column with experiment name
        experiment_id = os.path.basename(os.path.splitext(c)[0])
        df['experiment'] = experiment_id
        # if (np.sqrt(df['x_mm'].iloc[-1]**2 + df['y_mm'].iloc[-1]**2) < 50):
        # print('Experiment: {} was successful'.format(experiment_id))


        dfs.append(df)
        lengths.append(len(df.index))

    if ignore_idx:
        dfs = pd.concat(dfs, axis=0, ignore_index=True)

    else:
        dfs = pd.concat(dfs, axis=0, ignore_index=False)

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