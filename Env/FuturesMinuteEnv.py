# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:22:17 2019

@author: feifanhe
"""
import pandas as pd
import numpy as np
from collections import deque
import datetime
import time
import FuturesEnv

class Env(FuturesEnv.Env):
    # Constants
    CONTRACT = ['TX01', 'TX02', 'MTX01', 'MTX02']
    CONTRACT_IDX = {j:i for i, j in enumerate(CONTRACT)}
    CONTRACT_COUNT = len(CONTRACT)
    CONTRACT_SIZE = np.array([200, 200, 50, 50], dtype = int)
    MINUTES_PER_DAY = 300
    
    def __init__(self, futures_folder):
        super().__init__(futures_folder)
        
    def load_price(self):
        # create table
        open_price = pd.DataFrame(columns=self.CONTRACT)
        close_price = pd.DataFrame(columns=self.CONTRACT)
        
        # TAIEX
        tx01 = self.load_price_csv(self.futures_folder + 'tx01.csv')
        tx02 = self.load_price_csv(self.futures_folder + 'tx02.csv')
        mtx01 = self.load_price_csv(self.futures_folder + 'mtx01.csv')
        mtx02 = self.load_price_csv(self.futures_folder + 'mtx02.csv') 
        
        open_price['TX01'] = tx01['Open']
        open_price['TX02'] = tx02['Open']
        close_price['TX01'] = tx01['Close']
        close_price['TX02'] = tx02['Close']
        open_price['MTX01'] = mtx01['Open']
        open_price['MTX02'] = mtx02['Open']
        close_price['MTX01'] = mtx01['Close']
        close_price['MTX02'] = mtx02['Close']
        
        self.trading_time = pd.DatetimeIndex(tx01['Time'])
        self.open = open_price.values
        self.close = close_price.values
       
    def load_price_csv(self, filename):
        df = pd.read_csv(filename, index_col = 0)
        df['Time'] = pd.to_datetime(df['Time'])
        start_index = df[df['Time'] == self.start_time].index[0]
        head_index = start_index - self.history_steps
        end_index = start_index + self.steps
        df = df.iloc[head_index:end_index]
        return df
         
    def load_margin(self):
        df = pd.read_csv(self.futures_folder + 'margin.csv') #保證金
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])
        
        self.margin = pd.DataFrame(index = self.trading_time, columns = self.MARGIN_TYPE)
        for i, row in df.iterrows():
            for date in self.margin.loc[row['start']:row['end'] + pd.Timedelta('1 days')].index:
                self.margin.loc[date, self.MARGIN_TYPE] = row[self.MARGIN_TYPE]
        
        self.margin_ori = pd.DataFrame(index = self.trading_time, columns = self.CONTRACT)
        self.margin_maint = pd.DataFrame(index = self.trading_time, columns = self.CONTRACT)
        
        self.margin_ori[self.CONTRACT] = self.margin[['tx_ori', 'tx_ori', 'mtx_ori', 'mtx_ori']]
        self.margin_ori = self.margin_ori.values
        self.margin_maint[self.CONTRACT] = self.margin[['tx_maint', 'tx_maint', 'mtx_maint', 'mtx_maint']]
        self.margin_maint = self.margin_maint.values
        
    def load_settlement_price(self):
        self.settlement_price = pd.read_csv(self.futures_folder + 'settlement.csv')
        self.settlement_price['Date'] = pd.to_datetime(self.settlement_price['Date']).dt.date
        self.settlement_price = self.settlement_price.set_index('Date')
        
    def reset(
            self, 
            cash,
            start_time, 
            steps,
            history_steps):
        self.start_time = start_time
        self.start_date = pd.to_datetime(start_time).date().strftime('%Y-%m-%d')
        self.cash = cash
        self.steps = steps
        self.history_steps = history_steps
        
        self.cnt = 0
        self.pool = 0
        self.position = np.zeros(self.CONTRACT_COUNT, dtype=int)
        self.margin_ori_level = 0
        self.position_queue = [deque([]) for _ in range(self.CONTRACT_COUNT)]
        self.margin_call = 0
        
        #self.load_trading_day()
        self.load_price()
        self.load_margin()
        self.load_settlement_price()
        
        self.done = False

    def __settlement(self, time_index):
        # 結算價
        final_price = self.settlement_price.loc[self.trading_time[time_index].date(), 'Price']
        
        # 庫存點位
        position_point = np.zeros(self.CONTRACT_COUNT)
        for i in [0, 2]:
            position_point[i] = sum(self.position_queue[i]) * np.sign(self.position[i])
        
        # 結算點位
        settlement_point = self.position * final_price
        settlement_point[[1, 3]] = 0
        profit = np.sum((settlement_point - position_point) * self.CONTRACT_SIZE)
        self.pool += profit
        
        # 轉倉
        self.position[[0, 2]] = self.position[[1, 3]]
        self.position[[1, 3]] = 0
        self.position_queue[0] = self.position_queue[1].copy()
        self.position_queue[2] = self.position_queue[3].copy()
        self.position_queue[1].clear()
        self.position_queue[3].clear()
        
        # 調整保證金水位
        self.margin_ori_level = np.sum(self.margin_ori[time_index] * np.abs(self.position))
        
        return profit
    
    def step(self, action):
        time_index = self.cnt + self.history_steps
        profit = 0
        
        # 追繳保證金
        deal_liq = np.zeros(self.CONTRACT_COUNT, dtype = int)
        if self.margin_call > 0:
            if self.cash < self.margin_call:
                # liquidate
                cond_liq = self.position != 0
                profit_liq, deal_liq = self.__close(self.position * -1, cond_liq, self.open[time_index], self.margin_ori[time_index])
                profit += profit_liq
            else:
               self.cash -= self.margin_call
               self.pool += self.margin_call
               self.margin_call = 0
        
        # 委託單
        order = np.zeros(self.CONTRACT_COUNT, dtype = int)
        for symbol, volume in action:
            order[self.CONTRACT_IDX[symbol]] = volume
        order_original = order.copy()
        
        # 先平倉
        cond_close = (order * self.position) < 0
        profit_close, deal_close = self.__close(order, cond_close, self.open[time_index], self.margin_ori[time_index])
        profit += profit_close
        order -= deal_close
        
        # 建倉 / 新倉
        cond_new = (order * self.position) >= 0
        deal_new = self.__new(order, cond_new, self.open[time_index], self.margin_ori[time_index])
        
        order_deal = deal_liq + deal_close + deal_new
        
        # 結算
        if (self.trading_time[time_index].time() == datetime.time(13, 44) and 
            self.trading_time[time_index].date() in self.settlement_price.index):
            profit += self.__settlement(time_index)
            
        # 庫存點位
        position_point = np.zeros(self.CONTRACT_COUNT)
        for i in range(self.CONTRACT_COUNT):
            position_point[i] = sum(self.position_queue[i])
        position_point *= np.sign(self.position)
        
        # average cost
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_cost = np.nan_to_num(position_point / self.position)
        
        # 未實現損益
        unrealized = np.sum((self.close[time_index] * self.position - position_point) * self.CONTRACT_SIZE)
        
        # 檢查保證金水位
        margin_maint_level = np.sum(self.margin_maint[time_index] * np.abs(self.position))
        if self.pool + unrealized < margin_maint_level:
            self.margin_call = self.margin_ori_level - (self.pool + unrealized)
        
        self.cnt += 1
        if self.cnt == self.steps:
            self.done = True
        
        return self.cash, self.pool, unrealized, profit, self.position, avg_cost, order_original, order_deal, self.margin_call

#%%
if __name__ == '__main__':
    
    futures_folder = './futures_data_minute/'
    env = Env(futures_folder)
    
    cash = int(1e+6)
    start_time = '2016-01-19 09:00:00'
    steps = 5
    history_steps = 10
    
    env.reset(cash,
              start_time, 
              steps,
              history_steps)
    
    action = list([
            [['TX01',-1],['TX02',-1]],
            [['TX01',-1],['TX02',1]],
            [['TX01',3],['TX02',3]],
            [],
            [['TX01',-3],['TX02',-3]],
            ])
    
    for i in range(steps):
        print(f'[step {i+1}]')
        start_timer = time.time()
        cash, pool, unrealized, profit, position, avg_cost, order, deal, margin_call = env.step(action[i])
        end_timer = time.time()
        print('Order:\t', order)
        print('Deal:\t', deal)
        print('Position:\t', position)
        print('Avg. cost:\t', avg_cost)
        print('Profit \t Unrealized \t Margin Call')
        print(profit, '\t', unrealized, '\t', margin_call)
        print('Cash remains:', cash)
        print('Pool remains:', pool)
        print(f'[Time: {(end_timer - start_timer) * 1000}ms]')
        print()
        