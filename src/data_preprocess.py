#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from tqdm import tqdm 
import pickle
import argparse
import os

def transfrom_daily(df):
    data_dict = {}
    # split by agent
    for UserId, agent_group in tqdm(df.groupby('UserId')):
        # sort by datetime
        agent_group = agent_group.sort_values(by='CheckinTime')

        agent_dict = {}
        # split by days
        for date, group in agent_group.groupby(agent_group.CheckinTime.dt.date):
            date_dict = {
                'type': group.VenueType.values,
                'coor': group.loc[:, ["X","Y"]].values,
                'dayofweek': group.iloc[0].CheckinTime.dayofweek,
            }
            agent_dict[str(date)] = date_dict
        data_dict[UserId] = agent_dict
    return data_dict

def transfrom_by_period(df, time_period):
    data_dict = {}
    # split by agent
    for UserId, agent_group in tqdm(df.groupby('UserId')):
        # sort by datetime
        agent_group = agent_group.sort_values(by='CheckinTime')

        agent_dict = {}
        # split by weeks
        for week, group in agent_group.groupby('week'):
            date_dict = {
                'type': group.VenueType.values,
                'coor': group.loc[:, ["X","Y"]].values,
            }
            agent_dict['week_'+str(week)] = date_dict
        
        data_dict[UserId] = agent_dict
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--split_point', type=str, default="209-06-07T23:55:00.000")
    args = parser.parse_args()
    if 'Checkin.tsv' not in os.listdir(args.data_path):
        raise Exception(f'No Checkin.tsv can be found in {args.data_path}')
    if args.data_path == './data/geolife':
        df = pd.read_csv(os.path.join(args.data_path, 'Checkin.tsv'), sep=' ')
        df['CheckinTime'] = df['CheckinTime'].apply(lambda x: 'T'.join(x.split(',')))
    else:
        df = pd.read_csv(os.path.join(args.data_path, 'Checkin.tsv'), sep='\t')
    df['CheckinTime'] = pd.to_datetime(df['CheckinTime'])
    start_time = df.CheckinTime.min()
    time_period = 7
    end_time = start_time + pd.DateOffset(time_period)
    i = 0
    # split by time periods
    while start_time <= df.CheckinTime.max():
        flag = (df.CheckinTime >= start_time) & (df.CheckinTime < end_time)
        df.loc[flag, 'week'] = int(i)
        start_time = end_time
        end_time = start_time + pd.DateOffset(time_period)
        i += 1
    df.week = df.week.astype(int)
    df_train = df#.loc[df.CheckinTime < pd.to_datetime("2029-06-07T23:55:00.000")]
    df_test = df.loc[df.CheckinTime >= pd.to_datetime(args.split_point)]
    print('Start transform daily data.')
    data_dict_daily_train = transfrom_daily(df_train)
    data_dict_daily_test = transfrom_daily(df_test)
    print('Finish transform daily data.')
    max_len = 0
    min_len = 99
    for data_dict in [data_dict_daily_train, data_dict_daily_test]:
        for item in data_dict:
            for day in data_dict[item]:
                if len(data_dict[item][day]['type']) > max_len:
                    max_len = len(data_dict[item][day]['type'])
                if len(data_dict[item][day]['type']) < min_len:
                    min_len = len(data_dict[item][day]['type'])

    with open(os.path.join(args.data_path, 'data_dict_daily_train.pkl'), 'wb') as fp:
        pickle.dump(data_dict_daily_train, fp)
    with open(os.path.join(args.data_path, 'data_dict_daily_test.pkl'), 'wb') as fp:
        pickle.dump(data_dict_daily_test, fp)
        
    print('Start transform weekly data.')
    data_dict_weekly_train = transfrom_by_period(df_train, time_period=7)
    data_dict_weekly_test = transfrom_by_period(df_test, time_period=7)
    print('Finish transform weekly data.')

    max_len = 0
    min_len = 999
    for data_dict in [data_dict_weekly_train, data_dict_weekly_test]:
        for item in data_dict:
            for day in data_dict[item]:
                if len(data_dict[item][day]['type']) > max_len:
                    max_len = len(data_dict[item][day]['type'])
                if len(data_dict[item][day]['type']) < min_len:
                    min_len = len(data_dict[item][day]['type'])

    with open(os.path.join(args.data_path, 'data_dict_weekly_train.pkl'), 'wb') as fp:
        pickle.dump(data_dict_weekly_train, fp)
    with open(os.path.join(args.data_path, 'data_dict_weekly_test.pkl'), 'wb') as fp:
        pickle.dump(data_dict_weekly_test, fp)


    train_dates = list(df_train.CheckinTime.dt.date.apply(lambda x: str(x)).unique())
    test_dates = list(df_test.CheckinTime.dt.date.apply(lambda x: str(x)).unique())

    with open(os.path.join(args.data_path, 'dates_train.pkl'), 'wb') as fp:
        pickle.dump(train_dates, fp)
    with open(os.path.join(args.data_path, 'dates_test.pkl'), 'wb') as fp:
        pickle.dump(test_dates, fp)

