#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:29:21 2019

@author: feifanhe
"""

import os
import numpy as np
import pandas as pd

data_folder = './TW_daily/'
stocks = os.listdir(data_folder)
years = ['2016', '2017', '2018']
months = np.arange(1, 13)

df = pd.DataFrame([])

write_out = './TW_year/'

for i in range(len(stocks)):
    if len(os.listdir(data_folder + stocks[i])) < 12 * len(years): continue
    for j in range(len(years)):
        df = pd.DataFrame([])
        for k in range(len(months)):
            file_path = data_folder + stocks[i] + '/' + years[j] + str('%02d' % months[k]) + '.csv'
            tmp = pd.read_csv(file_path)
            df = df.append(tmp)
        
        df = df.reset_index().drop(['index'], axis=1)
        output_folder= write_out + stocks[i] + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print('Writing file:', output_folder + years[j] + '.xlsx')
        df.to_excel(output_folder + years[j] + '.xlsx')