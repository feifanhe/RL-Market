# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:22:17 2019

@author: feifanhe
"""
import pandas as pd
import numpy as np
from collections import deque

class Env:
    # Constants
    CONTRACT = ['TX01', 'TX02', 'MTX01', 'MTX02']
    CONTRACT_IDX = {j:i for i, j in enumerate(CONTRACT)}
    CONTRACT_COUNT = len(CONTRACT)
    CONTRACT_SIZE = np.array([200, 200, 50, 50], dtype = int)
    
    def __init__(self, futures_folder):
        # Env parameter initial
        self.futures_folder = futures_folder
        
    # 讀取台股交易日
    def load_trading_day(self):
        df = pd.read_excel('./stock_data/Y9999.xlsx')
        start_date_row = df.loc[df['年月日'] == self.start_date]
        assert len(start_date_row) > 0, '起始日無交易'
        start_date_index = start_date_row.index[0]
        assert start_date_index >= self.history_steps, '交易日資料不足'
        head_date_index = start_date_index - self.history_steps
        end_date_index = start_date_index + self.steps
        self.trading_day = pd.DatetimeIndex(df['年月日'].iloc[head_date_index:end_date_index])
        
    def load_price(self):
        # create table
        open_price = pd.DataFrame(index=self.trading_day, columns=self.CONTRACT)
        close_price = pd.DataFrame(index=self.trading_day, columns=self.CONTRACT)
        
        # TAIEX
        tx = pd.read_csv(self.futures_folder + 'tx_2016_2018_new.csv', index_col = 0)
        tx['Date'] = pd.to_datetime(tx['Date'])
        tx['near_month'] = tx.groupby('Date')['Contract'].rank()
        group_tx = tx.groupby('near_month')
        tx01 = group_tx.get_group(1).set_index('Date')
        tx02 = group_tx.get_group(2).set_index('Date')
        
        open_price['TX01'] = tx01['Open']
        open_price['TX02'] = tx02['Open']
        close_price['TX01'] = tx01['Close']
        close_price['TX02'] = tx02['Close']
        
        # Mini-TAIEX
        mtx = pd.read_csv(self.futures_folder + 'mtx_2016_2018_new.csv', index_col = 0)
        mtx['Date'] = pd.to_datetime(mtx['Date'])
        mtx['near_month'] = mtx.groupby('Date')['Contract'].rank()
        group_mtx = mtx.groupby('near_month')
        mtx01 = group_mtx.get_group(1).set_index('Date')
        mtx02 = group_mtx.get_group(2).set_index('Date')
        
        open_price['MTX01'] = mtx01['Open']
        open_price['MTX02'] = mtx02['Open']
        close_price['MTX01'] = mtx01['Close']
        close_price['MTX02'] = mtx02['Close']
        
        self.open = open_price.values
        self.close = close_price.values
        
    def load_margin(self):
        self.margin = pd.read_csv(self.futures_folder + 'margin.csv') #保證金
        self.margin['start'] = pd.to_datetime(self.margin['start'])
        self.margin['end'] = pd.to_datetime(self.margin['end'])
        
    def load_settlement_price(self):
        self.settlement_price = pd.read_csv(self.futures_folder + 'settlement.csv')
        self.settlement_price['Date'] = pd.to_datetime(self.settlement_price['Date'])
        self.settlement_price = self.settlement_price.set_index('Date')
    
    def reset(
            self, 
            cash,
            start_date, 
            steps,
            history_steps):
        self.start_date = start_date
        self.cash = cash
        self.steps = steps
        self.history_steps = history_steps

        self.cnt = 0
        self.pool = 0
        self.position = np.zeros(self.CONTRACT_COUNT, dtype=int)
        self.margin_ori_level = 0
        self.position_queue = [deque([]) for _ in range(self.CONTRACT_COUNT)]
        self.margin_call = 0
        
        self.load_trading_day()
        self.load_price()
        self.load_margin()
        self.load_settlement_price()
    
    def __update_margin(self, date_index):
        for index, row in self.margin.iterrows():
            if row['start'] <= self.trading_day[date_index] <= row['end']:
                self.margin_ori = np.array(
                        [row['tx_original']] * 2 + [row['mtx_original']] * 2)
                self.margin_maint = np.array(
                        [row['tx_maintenance']] * 2 + [row['mtx_maintenance']] * 2)
                return
        
    def __new(self, order, cond, open_price):
        deal_new = order.copy()
        deal_new[np.logical_not(cond)] = 0
        volume = np.abs(deal_new)
        margin = np.sum(self.margin_ori * volume)
        
        # evaluate the required original margin
        if self.pool < (margin + self.margin_ori_level):
            diff = margin + self.margin_ori_level - self.pool
            if diff > self.cash:
                tmp_cash = self.cash + self.pool - self.margin_ori_level
                for i in np.where(cond)[0]:
                    if (tmp_cash / self.margin_ori[i]) < volume[i]:
                        # 現金不足，計算最大可買張數
                        volume[i] = int(tmp_cash / self.margin_ori[i])
                    tmp_cash -= self.margin_ori[i] * volume[i]
                deal_new = np.sign(deal_new) * volume
                margin = np.sum((self.margin_ori * volume)[cond])
                diff = margin + self.margin_ori_level - self.pool
            self.pool += diff
            self.cash -= diff
            
        # append to position queue
        for i in np.where(cond)[0]:
            self.position_queue[i].extend([open_price[i]] * volume[i])

        self.position += deal_new
        self.margin_ori_level += margin
        
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
        position_point = np.zeros(self.CONTRACT_COUNT)
        for i in np.where(cond)[0]:
            for j in range(volume[i]):
                assert len(self.position_queue[i]) > 0
                position_point[i] += int(self.position_queue[i].popleft())
        position_point *= np.sign(self.position)
        
        profit = np.sum((close_point + position_point) * -1 * self.CONTRACT_SIZE)
        self.position += deal_close
        self.margin_ori_level -= np.sum(self.margin_ori[cond] * volume[cond])
        self.pool += profit
        
        return profit, deal_close
        
    def __settlement(self, date_index):
        # 結算價
        final_price = self.settlement_price.loc[self.trading_day[date_index], 'Price']
        
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
        self.margin_ori_level = np.sum(self.margin_ori * np.abs(self.position))
        
        return profit
    
    def step(self, action):
        date_index = self.cnt + self.history_steps
        self.__update_margin(date_index)
        profit = 0
        
        # 追繳保證金
        deal_liq = np.zeros(self.CONTRACT_COUNT, dtype = int)
        if self.margin_call > 0:
            if self.cash < self.margin_call:
                # liquidate
                cond_liq = self.position != 0
                profit_liq, deal_liq = self.__close(self.position * -1, cond_liq, self.open[date_index])
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
        cond_close = (order * self.position < 0)
        profit_close, deal_close = self.__close(order, cond_close, self.open[date_index])
        profit += profit_close
        order -= deal_close
        
        # 建倉 / 新倉
        cond_new = (order * self.position >= 0)
        deal_new = self.__new(order, cond_new, self.open[date_index])
        
        order_deal = deal_liq + deal_close + deal_new
        
        # 結算
        if self.trading_day[date_index] in self.settlement_price.index:
            profit += self.__settlement(date_index)
            
        # 庫存點位
        position_point = np.zeros(self.CONTRACT_COUNT)
        for i in range(self.CONTRACT_COUNT):
            position_point[i] = sum(self.position_queue[i])
        position_point *= np.sign(self.position)
        
        # average cost
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_cost = np.nan_to_num(position_point / self.position)
        
        # 未實現損益
        unrealized = np.sum((self.close[date_index] * self.position - position_point) * self.CONTRACT_SIZE)
        
        # 檢查保證金水位
        margin_maint_level = np.sum(self.margin_maint * np.abs(self.position))
        if self.pool + unrealized < margin_maint_level:
            self.margin_call = self.margin_ori_level - (self.pool + unrealized)
        
        self.cnt += 1
        
        return self.cash, self.pool, unrealized, profit, self.position, avg_cost, order_original, order_deal, self.margin_call

#%%
if __name__ == '__main__':
    
    futures_folder = './futures_data/'
    env = Env(futures_folder)
    
    cash = int(1e+6)
    start_date = '2016/01/19'
    steps = 5
    history_steps = 10
    
    env.reset(cash,
              start_date, 
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
        cash, pool, unrealized, profit, position, avg_cost, order, deal, margin_call = env.step(action[i])
        print('Order:\t', order)
        print('Deal:\t', deal)
        print('Position:\t', position)
        print('Avg. cost:\t', avg_cost)
        print('Profit \t Unrealized \t Margin Call')
        print(profit, '\t', unrealized, '\t', margin_call)
        print('Cash remains:', cash)
        print('Pool remains:', pool)
        print()
        