# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:22:17 2019

@author: feifanhe
"""
import pandas as pd
import numpy as np
from collections import deque

class Env:
    def __init__(self, futures_folder):
        # Env parameter initial
        self.futures_folder = futures_folder
        self.target_idx = {j:i for i, j in enumerate(self.TARGET)}
        
    # Constants
    TARGET = ['TX01', 'TX02', 'MTX01', 'MTX02']
    TARGET_COUNT = len(TARGET)
    
    # 讀取台股交易日
    def load_trading_day(self):
        df = pd.read_excel('./stock_data/Y9999.xlsx')
        start_date_row = df.loc[df['年月日'] == self.start_date]
        assert len(start_date_row) > 0, '起始日無交易'
        start_date_index = start_date_row.index[0]
        assert start_date_index >= self.history_steps, '交易日資料不足'
        head_date_index = start_date_index - self.history_steps
        #end_date_index = start_date_index + self.steps
        #self.trading_day = pd.DatetimeIndex(df['年月日'].iloc[head_date_index:end_date_index])
        self.trading_day = pd.DatetimeIndex(df['年月日'].iloc[head_date_index:])
        #self.trading_day = pd.DatetimeIndex(df['年月日'])
        
    def load_price(self):
        self.open = pd.DataFrame(index=self.trading_day, columns=self.TARGET)
        self.close = pd.DataFrame(index=self.trading_day, columns=self.TARGET)
        
        #大台dataset
        data = pd.read_csv(self.futures_folder + 'tx_2016_2018_new.csv', index_col = 0)
        data['Date'] = pd.to_datetime(data['Date'])
        data['contract_rank'] = data.groupby('Date')['Contract'].rank()
        group_data = data.groupby('contract_rank')
        data_tx01 = group_data.get_group(1).set_index('Date')
        data_tx02 = group_data.get_group(2).set_index('Date')
        
        self.open['TX01'] = data_tx01['Open']
        self.open['TX02'] = data_tx02['Open']
        self.close['TX01'] = data_tx01['Close']
        self.close['TX02'] = data_tx02['Close']
        
        #小台dataset
        mdata = pd.read_csv(self.futures_folder + 'mtx_2016_2018_new.csv', index_col = 0)
        mdata['Date'] = pd.to_datetime(mdata['Date'])
        mdata['contract_rank']=mdata.groupby('Date')['Contract'].rank()
        group_mdata = mdata.groupby('contract_rank')
        data_mtx01 = group_mdata.get_group(1).set_index('Date')
        data_mtx02 = group_mdata.get_group(2).set_index('Date')
        
        self.open['MTX01'] = data_mtx01['Open']
        self.open['MTX02'] = data_mtx02['Open']
        self.close['MTX01'] = data_mtx01['Close']
        self.close['MTX02'] = data_mtx02['Close']
        
        self.open = self.open.values
        self.close = self.close.values
        
    def load_due_date(self):
        #到期日dataset
        self.due_date = pd.read_csv(self.futures_folder + 'DueDate.csv')
        self.due_date['dueDate'] = pd.to_datetime(self.due_date['dueDate'])
        self.due_date = self.due_date.set_index('dueDate')
        
    def load_maintain_margin(self):
        self.maintain_margin = pd.read_csv(self.futures_folder + 'maintain_margin.csv') #保證金
        self.maintain_margin['start'] = pd.to_datetime(self.maintain_margin['start'])
        self.maintain_margin['end'] = pd.to_datetime(self.maintain_margin['end'])
    
    def get_margin(self, date_index):
        for index, row in self.maintain_margin.iterrows():
            if row['start'] <= self.trading_day[date_index] <= row['end']:
                self.margin_original = np.array(
                        [row['tx_original'],
                         row['tx_original'],
                         row['mtx_original'],
                         row['mtx_original']])
                self.margin_maintenance = np.array(
                        [row['tx_maintenance'],
                         row['tx_maintenance'],
                         row['mtx_maintenance'],
                         row['mtx_maintenance']])
                return
    
    def reset(
            self, 
            start_date, 
            cash,
            steps,
            history_steps):
        self.start_date = start_date
        self.cash = cash
        self.steps = steps
        self.history_steps = history_steps

        self.cnt = 0
        self.pool = 0
        self.position = np.zeros(self.TARGET_COUNT, dtype=int)
        self.position_margin_original = 0
        self.position_queue = [deque([]) for _ in range(self.TARGET_COUNT)]
        self.unit_price = np.array([200, 200, 50, 50], dtype = int)
        self.margin_call = 0
        
        self.load_trading_day()
        self.load_price()
        self.load_due_date()
        self.load_maintain_margin()
        
    def __new(self, order, cond, open_price):
        deal_new = order.copy()
        deal_new[np.logical_not(cond)] = 0
        volume = np.abs(deal_new)
        margin = np.sum(self.margin_original * volume)
        
        # evaluate the required original margin
        if self.pool < (margin + self.position_margin_original):
            diff = margin + self.position_margin_original - self.pool
            if diff > self.cash:
                tmp_cash = self.cash + self.pool - self.position_margin_original
                for i in np.where(cond)[0]:
                    if (tmp_cash / self.margin_original[i]) < volume[i]:
                        # 現金不足，計算最大可買張數
                        volume[i] = int(tmp_cash / self.margin_original[i])
                    tmp_cash -= self.margin_original[i] * volume[i]
                deal_new = np.sign(deal_new) * volume
                margin = np.sum((self.margin_original * volume)[cond])
                diff = margin + self.position_margin_original - self.pool
            self.pool += diff
            self.cash -= diff
            
        # append to position queue
        for i in np.where(cond)[0]:
            self.position_queue[i].extend([open_price[i]] * volume[i])

        self.position += deal_new
        self.position_margin_original += margin
        
        return deal_new
                    
    def __close(self, order, cond, open_price):
        deal_close = order.copy()
        deal_close[np.logical_not(cond)] = 0

        volume = np.abs(deal_close)
        position_volume = np.abs(self.position)

        # 平倉量超出庫存
        cond_over_sell = (cond & (volume > position_volume))
        deal_close[cond_over_sell] = position_volume[cond_over_sell] * -1
        volume = np.abs(deal_close)
        
        # 平倉點位
        close_point = open_price * deal_close
        
        # 庫存點位
        position_point = np.zeros(self.TARGET_COUNT)
        for i in np.where(cond)[0]:
            for j in range(volume[i]):
                assert len(self.position_queue[i]) > 0
                position_point[i] += int(self.position_queue[i].popleft())
        position_point *= np.sign(self.position)
        
        profit = np.sum((close_point + position_point) * -1 * self.unit_price)
        self.position += deal_close
        self.position_margin_original -= np.sum(self.margin_original[cond] * volume[cond])
        self.pool += profit
        
        return profit, deal_close
        
    def __settlement(self, due_point):
        # 庫存點位
        position_point = np.zeros(self.TARGET_COUNT)
        for i in [0, 2]:
            position_point[i] = sum(self.position_queue[i])
        position_point *= np.sign(self.position)
        
        # 結算點位
        settlement_point = self.position * due_point
        settlement_point[[1, 3]] = 0
        profit = np.sum((settlement_point - position_point) * self.unit_price)
        
        # 轉倉
        self.position[[0, 2]] = self.position[[1, 3]]
        self.position[[1, 3]] = 0
        self.position_queue[0] = self.position_queue[1].copy()
        self.position_queue[2] = self.position_queue[3].copy()
        self.position_queue[1].clear()
        self.position_queue[3].clear()
        
        # 調整保證金水位
        self.position_margin_original = np.sum(self.margin_original * np.abs(self.position))
        
        return profit
        
    def step(self, action):
        date_index = self.cnt
        self.get_margin(date_index)
        
        # 委託單
        order = np.zeros(self.TARGET_COUNT, dtype = int)
        for code, volume in action:
            order[self.target_idx[code]] = volume
        order_original = order.copy()
        
        # 先平倉
        cond_close = (order * self.position < 0)
        profit, deal_close = self.__close(order, cond_close, self.open[date_index])
        order -= deal_close
        
        # 建倉 / 新倉
        cond_new = (order * self.position >= 0)
        deal_new = self.__new(order, cond_new, self.open[date_index])
        
        order_deal = deal_close + deal_new
        
        # 結算
        if self.trading_day[date_index] in env.due_date.index:
            due_point = self.due_date.loc[self.trading_day[date_index], 'price']
            profit += self.__settlement(due_point)
            
        # 庫存點位
        position_point = np.zeros(self.TARGET_COUNT)
        for i in range(self.TARGET_COUNT):
            position_point[i] = sum(self.position_queue[i])
        position_point *= np.sign(self.position)
        
        # average cost
        avg_cost = np.nan_to_num(position_point / self.position)
        
        # 未實現損益
        unrealized = np.sum((self.close[date_index] * self.position - position_point) * self.unit_price)
        
        # 檢查保證金水位
        position_margin_maintenance = np.sum(self.margin_maintenance * np.abs(self.position))
        if self.pool + unrealized < position_margin_maintenance:
            self.margin_call = self.position_margin_original - (self.pool + unrealized)
        
        self.cnt += 1
        
        return self.cash, unrealized, profit, self.position, avg_cost, order_original, order_deal, self.margin_call

#%%
if __name__ == '__main__':
    
    futures_folder = './futures_data/'
    env = Env(futures_folder)
    
    cash = 1e+6
    start_date = '2016/01/19'
    steps = 3
    history_steps = 0
    
    env.reset(start_date, 
              cash,
              steps,
              history_steps)
    
    action = list([
            [['TX01',1],['TX02',1]],
            [['TX01',2],['TX02',1]],
            [['TX01',-3],['TX02',-3]],            
            ])
    
    
    for i in range(steps):
        cash, unrealized, profit, position, avg_cost, order, deal, margin_call = env.step(action[i])
        print('Cash remains:', cash)
        print('Profit \t Unrealized \t Margin Call')
        print(profit, '\t', unrealized, '\t', margin_call)
        print('Position:\t', position)
        print('Avg. cost:\t', avg_cost)
        print('Order:\t', order)
        print('Deal:\t', deal)
        print()
        # if margin_call > 0, and, cash < margin_call in next step, force liquidating?
        