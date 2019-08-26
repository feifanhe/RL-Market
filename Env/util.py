#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:22:15 2019

@author: feifanhe
"""

import pandas as pd

dfs = pd.DataFrame([])
for year in range(2018, 2019):
    df = pd.read_csv(f'./futures_data_minute/tx_{year}_min.csv', index_col = 0)
    dfs = dfs.append(df)
    
dfs[['Open', 'High', 'Low', 'Close', 'Volume']] = dfs[['Open', 'High', 'Low', 'Close', 'Volume']].astype(int)
dfs['Date'] = pd.to_datetime(dfs['Date'])
dfs = dfs.reset_index().drop('index', axis=1)

#%%
dfs['near_month'] = dfs.groupby(['Date', 'Time'])['Contract'].rank()

#%%
group_dfs = dfs.groupby('near_month')
tx1 = group_dfs.get_group(1)
tx2 = group_dfs.get_group(2)

#%%
tx1 = tx1.reset_index().drop(['index', 'near_month'], axis=1)
tx2 = tx2.reset_index().drop(['index', 'near_month'], axis=1)

#tx1.to_csv('./futures_data_minute/mtx01_min.csv')
#tx2.to_csv('./futures_data_minute/mtx02_min.csv')

"""
台指近2分鐘資料少8天：
{Timestamp('2017-11-30 00:00:00'),
 Timestamp('2017-12-21 00:00:00'),
 Timestamp('2017-12-22 00:00:00'),
 Timestamp('2017-12-25 00:00:00'),
 Timestamp('2017-12-26 00:00:00'),
 Timestamp('2017-12-27 00:00:00'),
 Timestamp('2017-12-28 00:00:00'),
 Timestamp('2017-12-29 00:00:00')}

小台指
{Timestamp('2016-11-30 00:00:00'),
 Timestamp('2017-11-30 00:00:00'),
 Timestamp('2017-12-21 00:00:00'),
 Timestamp('2017-12-22 00:00:00'),
 Timestamp('2017-12-25 00:00:00'),
 Timestamp('2017-12-26 00:00:00'),
 Timestamp('2017-12-27 00:00:00'),
 Timestamp('2017-12-28 00:00:00'),
 Timestamp('2017-12-29 00:00:00'),
 Timestamp('2018-11-30 00:00:00')}
    """