#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:26:55 2019

@author: feifanhe
"""

import requests
import datetime as dt
import pandas as pd
from io import StringIO
import sys


if len(sys.argv) < 3:
    print ('Need arguments: [start-date] [end-date]')
    exit
else:
    init_date = pd.to_datetime(sys.argv[1]).date()
    end_date = pd.to_datetime(sys.argv[2]).date()

    date = init_date    
    host = '140.113.73.133'   # 請填入資料庫所在主機的 IP 位址
    columns = ['Symbol', 'Contract', 'Con_CP', 'Con_SP', 'Date', 'Time',
               'Open', 'High', 'Low', 'Close', 'Volume']
    due_date = pd.DatetimeIndex(pd.read_csv('DueDate_txo.csv')['dueDate'])

    while date <= end_date:
        next_date = date + dt.timedelta(days = 1)
        tf = 60
        url = 'http://{}:5984/market/_design/options/_list/kbar/txo?startkey=[\"{}\"]&endkey=[\"{}\"]&tf={}'.format(host, date, next_date, tf)
        print('Downloading [{}, {}]...'.format(date, next_date))
        res = requests.get(url)
        
        print('Reading data...')
        df = pd.read_csv(StringIO(res.text), names=columns).drop(0)
        
        print('Raw data rows:', df.size)
        if df.size == 0 :
            date = next_date
            continue
        
        print('Extracting data...')
        group_df = df.groupby('Contract')
        year = date.year
        month = date.month
        due = due_date[(year - 2016) * 12 + month - 1]
        if date.day > due.day:
            month += 1
        for i in range(2):
            if month > 12:
                year += 1
                month -= 12
            contract = '%04d%02d' % (year, month + i)
            print('Target contract:', contract)
            df_out = group_df.get_group(contract).reset_index().drop('index', axis=1)
            print('Extracted data rows:', df_out.size)
    
            filename = './{}/{}-{}.csv'.format(date.year, date, i + 1)
            print('Writing file [{}]...'.format(filename))
            df_out.to_csv(filename)
        date = next_date
        
