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
    parser.add_argument('--data_path', type=str, default='./data/geolife/')
    parser.add_argument('--dataset_name', type=str, default='20-needles-69-agents-0.8')
    args = parser.parse_args()
    data_path = args.data_path
    dataset_name = args.dataset_name
    df_train = pd.read_csv(os.path.join(data_path, 'needles-train-test', f'train-{dataset_name}-normal-portion.tsv'), sep=' ')
    df_test = pd.read_csv(os.path.join(data_path, 'needles-train-test', f'test-{dataset_name}-normal-portion.tsv'), sep=' ')
    save_dir = os.path.join(data_path, dataset_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    for df in [df_train, df_test]:
        df['CheckinTime'] = df['ArrivingTime'].apply(lambda x: 'T'.join(x.split(',')))
        df['CheckinTime'] = pd.to_datetime(df['CheckinTime'])
        df['CheckinDate'] = df['CheckinTime'].apply(lambda x: x.date())
    df_train = df_train.rename(
        columns = {
            'Longitude': 'X',
            'Latitude': 'Y',
            'AgentID': 'UserId',
            'LocationType': 'VenueType',
        }
    )
    df_test = df_test.rename(
        columns = {
            'Longitude': 'X',
            'Latitude': 'Y',
            'AgentID': 'UserId',
            'LocationType': 'VenueType',
        }
    )
    start_time = df_train.CheckinTime.min()
    time_period = 7
    end_time = start_time + pd.DateOffset(time_period)
    i = 0
    # split by time periods
    while start_time <= df_train.CheckinTime.max():
        flag = (df_train.CheckinTime >= start_time) & (df_train.CheckinTime < end_time)
        df_train.loc[flag, 'week'] = int(i)
        start_time = end_time
        end_time = start_time + pd.DateOffset(time_period)
        i += 1
    df_train.week = df_train.week.astype(int)

    start_time = df_test.CheckinTime.min()
    time_period = 7
    end_time = start_time + pd.DateOffset(time_period)
    i = 0
    while start_time <= df_test.CheckinTime.max():
        flag = (df_test.CheckinTime >= start_time) & (df_test.CheckinTime < end_time)
        df_test.loc[flag, 'week'] = int(i)
        start_time = end_time
        end_time = start_time + pd.DateOffset(time_period)
        i += 1
    df_test.week = df_test.week.astype(int)



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

    with open(os.path.join(save_dir, 'data_dict_daily_train.pkl'), 'wb') as fp:
        pickle.dump(data_dict_daily_train, fp)
    with open(os.path.join(save_dir, 'data_dict_daily_test.pkl'), 'wb') as fp:
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

    with open(os.path.join(save_dir, 'data_dict_weekly_train.pkl'), 'wb') as fp:
        pickle.dump(data_dict_weekly_train, fp)
    with open(os.path.join(save_dir, 'data_dict_weekly_test.pkl'), 'wb') as fp:
        pickle.dump(data_dict_weekly_test, fp)


    train_dates = list(df_train.CheckinTime.dt.date.apply(lambda x: str(x)).unique())
    test_dates = list(df_test.CheckinTime.dt.date.apply(lambda x: str(x)).unique())

    with open(os.path.join(save_dir, 'dates_train.pkl'), 'wb') as fp:
        pickle.dump(train_dates, fp)
    with open(os.path.join(save_dir, 'dates_test.pkl'), 'wb') as fp:
        pickle.dump(test_dates, fp)

