#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 00:00:49 2019

@author: feifanhe
"""

import pandas as pd


df = pd.read_excel('Y9999.xlsx')
dates = pd.DatetimeIndex(df['年月日'])

df = pd.read_excel('Ex-Right.xlsx')
df['公司'] = df['公司'].str.split(' ', expand = True)[0].astype(int)
df = df.drop('股東會年度', axis=1)

target = sorted(list(set(df['公司'].values)))

data = pd.DataFrame(index = dates, columns = target).fillna(0)

for i, row in df.iterrows():
    data.loc[row['除權日(配股)'], row['公司']] = row['無償配股合計(元)']