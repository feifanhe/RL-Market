# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:44:12 2019

@author: Fei-fan He
"""
import pandas as pd
import numpy as np
from collections import deque
from itertools import product
import OptionEnv

class Env(OptionEnv.Env):
    def __init__(self, option_folder):
        super().__init__(option_folder)
        
    def load_trading_time(self):
        # 先用期貨價格資料
        global df, start_time_row
        df = pd.read_csv('./futures_data_minute/tx01.csv', index_col = 0)
        df['Time'] = pd.to_datetime(df['Time'])
        start_time_row = df.loc[df['Time'] == self.start_time]
        assert len(start_time_row) > 0, '無交易資料'
        start_time_index = start_time_row.index[0]
        assert start_time_index >= self.history_steps, '交易資料不足'
        head_time_index = start_time_index - self.history_steps
        end_time_index = start_time_index + self.steps
        self.taiex_open = df['Open'].iloc[head_time_index:end_time_index].values.astype(int)
        self.taiex_close = df['Close'].iloc[head_time_index:end_time_index].values.astype(int)
        self.trading_time = pd.DatetimeIndex(df['Time'].iloc[head_time_index:end_time_index])
        
    def load_settlement_price(self):
        self.settlement_price = pd.read_csv(self.option_folder + 'settlement.csv')
        self.settlement_price['Date'] = pd.to_datetime(self.settlement_price['Date']).dt.date
        self.settlement_price = self.settlement_price.set_index('Date')
        
    def load_margin(self):
        df = pd.read_csv(self.option_folder + 'margin.csv') #保證金
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])
        
        self.margin = pd.DataFrame(index = self.trading_time, columns = self.MARGIN_TYPE)
        for i, row in df.iterrows():
            for date in self.margin.loc[row['start']:row['end'] + pd.Timedelta('1 days')].index:
                self.margin.loc[date, self.MARGIN_TYPE] = row[self.MARGIN_TYPE]
        
        self.margin = self.margin.values
        
    def load_price(self):
        txo = [[pd.DataFrame() for _ in range(2)] for _ in range(2)]
        
        for date in np.unique(env.trading_time.date):
            for near_month in range(1, 3):
                df = pd.read_csv(f'{self.option_folder}{date.year}/{date}-{near_month}.csv', index_col = 0)
                df['Time'] = df['Date'] + ' ' + df['Time']
                df['Time'] = pd.to_datetime(df['Time'])
                df_C = df[df['Con_CP'] == 'C']
                df_P = df[df['Con_CP'] == 'P']
                txo[near_month - 1][0] = txo[near_month - 1][0].append(df_C)
                txo[near_month - 1][1] = txo[near_month - 1][1].append(df_P)
        
        self.sp = np.unique(np.concatenate(
                [txo[m][n]['Con_SP'].values
                for m, n in product(range(2), range(2))], axis=None))
        self.sp_cnt = len(self.sp)
        self.sp_idx = {j: i for i, j in enumerate(self.sp)}
        
        total_steps = self.steps + self.history_steps
        open_prem = np.zeros((total_steps, 2, 2, self.sp_cnt))
        close_prem = np.zeros((total_steps, 2, 2, self.sp_cnt))
        
        for m, n in product(range(2), range(2)):
            df = txo[m][n]
            date_grouped = df.groupby('Time')
            for s in range(total_steps):
                for _, row in date_grouped.get_group(self.trading_time[s]).iterrows():
                    sp = self.sp_idx[row['Con_SP']]
                    open_prem[s, m, n, sp] = row['Open']
                    close_prem[s, m, n, sp] = row['Close']
        
        self.open = open_prem
        self.close = close_prem
    
    def reset(self,
              cash,
              start_time,
              steps,
              history_steps):
        
        self.cash = cash
        self.start_time = start_time
        self.steps = steps
        self.history_steps = history_steps
        
        self.load_trading_time()
        self.load_settlement_price()
        self.load_margin()
        self.load_price()
        
        self.cnt = 0
        self.pool = 0
        self.margin_ori_lvl = 0
        self.margin_maint_lvl = 0
        self.margin_call = 0
        self.position = np.zeros((2, 2, self.sp_cnt), dtype=int)
        self.position_queue = [[[
                deque([]) 
                for _ in range(self.sp_cnt)]
                for _ in range(2)]
                for _ in range(2)]
        
        self.done = False
        
    def __settlement(self, time_index):
        # 履約價
        strike_point = self.position[0] * self.sp
        
        # 結算價
        final_price = self.settlement_price.loc[self.trading_time[time_index].date(), 'Price']
        
        # 結算點位
        settlement_point = self.position[0] * final_price
        
        profit = np.sum(np.maximum(settlement_point - strike_point, 0).T * [1, -1]) * self.MULTIPLIER
        self.pool += profit
        
        self.position[0] = 0
        position_point = np.zeros(self.position.shape)
        for m, n, i in product(range(2), range(2), range(self.sp_cnt)):
            position_point[m, n, i] = sum(self.position_queue[m][n][i]) * np.sign(self.position[m, n, i])
        close_point = (self.close[time_index] * self.position).astype(int)
        unrealized = int(np.sum(close_point - position_point)) * self.MULTIPLIER
        
        # 轉倉
        self.position[0] = self.position[1]
        self.position[1] = 0
        
        for n, o in product(range(2), range(self.sp_cnt)): 
            self.position_queue[0][n][o] = self.position_queue[1][n][o].copy()
            self.position_queue[1][n][o].clear()

        return profit, unrealized
    
    def step(self, action):
        time_index = self.cnt + self.history_steps
        self.__update_margin(time_index)

        profit = 0
        
        # 追繳保證金
        deal_liq = np.zeros(self.position.shape, dtype=int)
        if self.margin_call > 0:
            if self.cash < self.margin_call:
                # liquidate
                cond_liq = self.position < 0
                profit_liq, deal_liq = self.__close(
                        self.position * -1,
                        cond_liq,
                        self.open[time_index],
                        self.taiex_open[time_index])
                profit += profit_liq
            else:
               self.cash -= self.margin_call
               self.pool += self.margin_call
               self.margin_call = 0
        
        # 委託
        order = np.zeros(self.position.shape, dtype=int)
        for near_month, cp, sp, volume in action:
            order[self.CONTRACT_IDX[near_month], 
                  self.CP_IDX[cp],
                  self.sp_idx[sp]] = volume
        order_original = order.copy()
        
        # 先平倉
        cond_close = (order * self.position) < 0
        deal_close, profit_close = self.__close(order, 
                                                cond_close, 
                                                self.open[time_index], 
                                                self.taiex_open[time_index])
        order -= deal_close
        profit += profit_close
        
        # 建倉 / 新倉
        cond_new = np.logical_not(cond_close)
        deal_new, cost, premium = self.__new(order, 
                                             cond_new, 
                                             self.open[time_index],
                                             self.taiex_open[time_index])
        
        order_deal = deal_close + deal_new + deal_liq
        
        # 結算
        unrealized = 0
        if self.trading_time[time_index].date() in self.settlement_price.index:
            profit_sett, unrealized = self.__settlement(time_index)
            profit += profit_sett
        else:
            # 未實現損益
            position_point = np.zeros(self.position.shape)
            for m, n, i in product(range(2), range(2), range(self.sp_cnt)):
                position_point[m, n, i] = sum(self.position_queue[m][n][i]) * np.sign(self.position[m, n, i])
            close_point = self.close[time_index] * self.position
            unrealized = int(np.sum(close_point - position_point)) * self.MULTIPLIER
        
        # 檢查保證金水位
        self.__update_margin_lvl(self.taiex_close[time_index])
        if self.pool < self.margin_maint_lvl:
            # 低於維持保證金，追繳至原始保證金
            self.margin_call = self.margin_ori_lvl - self.pool
        
        self.cnt += 1
        if self.cnt == self.steps:
            self.done = True
            
        return (self.cash, 
                self.pool,
                cost,       # 權利金支出
                premium,    # 權利金收入
                unrealized, # 權利金差
                profit, 
                self.position, 
                order_original, 
                order_deal, 
                self.margin_call
                )
#%%    
if __name__ == '__main__':
    import time
    option_folder = './option_data_minute/'
    
    start_time = '2016/01/04 13:43:00'
    steps = 4
    action = list([
            [['TXO01', 'C', 7900, 1], ['TXO02', 'C', 7900, -2], ['TXO01', 'P', 7700, -1]],
            [['TXO02', 'P', 8700, 2], ['TXO01', 'C', 7900, 2], ['TXO01', 'P', 8500, 1]],
            [['TXO01', 'C', 9200, 1],['TXO02', 'C', 9200, -1]],
            [['TXO01', 'P', 7700, -1]]
            ])   
    
    cash = 1e+6
    env = Env(option_folder)  
    env.reset(cash, start_time, steps, 0)
    
    #%%
    
    total_time = 0
    for i in range(steps):
        print(f'[Step {i+1}]')
        print(action[i])
        

        start_time = time.time()
        cash, pool, cost, premium, unrealized, profit, position, order, deal, margin_call = env.step(action[i])
        end_time = time.time()
        
        print('Position:\n', position)
        print('Cost\tPremium\tProfit\tUnrl\tMargin Call')
        print(f'{cost}\t{premium}\t{profit}\t{unrealized}\t{margin_call}')
        print('Cash remains:', cash)
        print('Pool remains:', pool)
        
        step_time = (end_time - start_time) * 1000
        total_time += step_time
        print(f'[Time: {step_time}ms]')
        print()
    print(f'[Total time: {total_time}ms]')