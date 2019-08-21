# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:39:05 2019

@author: pshkre
"""

import requests
import pandas as pd
from io import StringIO

localhost = '140.113.73.133'   # 請填入資料庫所在主機的 IP 位址
start = '2018-01-01'
end = '2018-12-31'   # 資料正常為取到 t-1，settlement 因為有 bug 僅取到 t-2（要往後推2天）
#%%

# 1. 直接 request K Bar（包含複式合約、遠月合約）
# 1-1. Daily K Bar 開高低收量
'''
url = 'http://' + localhost + ':5984/market/_design/futures/_list/kbar/tx?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]'   # 大台指期
url = 'http://' + localhost + ':5984/market/_design/futures/_list/kbar/mtx?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]'   # 小台指期

url = 'http://' + localhost + ':5984/market/_design/options/_list/kbar/txo?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]'   # 台指選
'''
# 1-2. n sec K Bar, tf argument stands for timeframe
tf = 1   # 單位為秒 (sec)
#url = 'http://' + localhost + ':5984/market/_design/futures/_list/kbar/tx?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]&tf=' + str(tf)   # 大台指期
#url = 'http://' + localhost + ':5984/market/_design/futures/_list/kbar/mtx?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]&tf=' + str(tf)   # 小台指期
url = 'http://' + localhost + ':5984/market/_design/options/_list/kbar/txo?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]&tf=' + str(tf)   # 台指選

req = requests.get(url)
#settle_list = req.text[1:-1].split(',\n')[:-1]
#df = pd.read_csv(StringIO(req.text))
'''
# 2. 結合近月 settlement 取得近月 K bar
settle_url = 'http://' + localhost + ':5984/market/_design/futures/_list/settlement/tx?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]'   # 大台指期
settle_url = 'http://' + localhost + ':5984/market/_design/futures/_list/settlement/mtx?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]'   # 小台指期
settle_url = 'http://' + localhost + ':5984/market/_design/options/_list/settlement/txo?startkey=[\"' + start + '\"]&endkey=[\"' + end + '\"]'   # 台指選
settle_req = requests.get(settle_url)
settle_list = settle_req.text[1:-1].split(',\n')[:-1]
'''
df = pd.DataFrame([])
#%%
for key in req:
    '''
    # 2-1. 期貨
    url = 'http://' + localhost + ':5984/market/_design/futures/_list/kbar/tx?keys=[' + key + ']&tf=1'   # 秒資料
    req = requests.get(url)
    df = df.append(pd.read_csv(StringIO(req.text)), ignore_index=True)
    '''
    
    # 2-2. 選擇權
    symbol = 'C'   # 買權填C，賣權填P
    for K in range(5000, 15000, 100):   # 連續範圍履約價：這個範圍基本上會包含所有可能的履約價
        txo_key = key[:-1] + ',\"' + symbol + '\",\"' + str(K) + '\"]'
        url = 'http://' + localhost + ':5984/market/_design/options/_list/kbar/txo?keys=[' + txo_key + ']&tf=1'   # 秒資料
        req = requests.get(url)
        df = df.append(pd.read_csv(StringIO(req.text)), ignore_index=True)
#%%
df.to_pickle('data.pickle')   # 存檔，之後可以不用重新抓資料

#%%

data = pd.read_csv('txo_snd_2016.csv')



