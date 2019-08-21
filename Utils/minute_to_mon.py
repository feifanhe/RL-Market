#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:29:21 2019

@author: feifanhe
"""

import os
import pandas as pd

header = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'Pattern', 'IsReal']

data_folder = './TW_minute/'
write_out = './TW_minute_month/'
categories = sorted(os.listdir(data_folder))

for h in range(len(categories)):
    cur_dir = '%s%s/' % (data_folder, categories[h])
    stocks = sorted(os.listdir(cur_dir))
    years = ['2016', '2017', '2018']

    for i in range(len(stocks)):
        for j in range(len(years)):
            months = sorted(os.listdir('%s%s/%s/' % (cur_dir, stocks[i], years[j])))
            for k in range(len(months)):
                days = sorted(os.listdir('%s%s/%s/%s' % (cur_dir, stocks[i], years[j], months[k])))
                df = pd.DataFrame([])
                for l in range(len(days)):
                    filename = '%s%s/%s/%s/%s' % (cur_dir, stocks[i], years[j], months[k], days[l])
                    tmp = pd.read_csv(filename, sep=',', names=header).drop(0)
                    df = df.append(tmp)
                    #break
                df = df.reset_index().drop(['index'], axis=1)
                output_dir = '%s%s/%s/' % (write_out, stocks[i], years[j])
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_file = '%s%s.xlsx' % (output_dir, months[k])
                print('Writing file:', output_file)
                df.to_excel(output_file)