# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:22:17 2019

@author: feifanhe
"""
import pandas as pd
import numpy as np
from collections import deque
from itertools import product
import time

class Env:
    # Constants
    TAX_RATE = 0.001 #交易稅
    TAX_RATE_SETT = 0.00002 #交易稅(結算)
    FEE = 15 #手續費
    MULTIPLIER = 50
    CONTRACT_IDX = {'TXO01': 0, 'TXO02': 1}
    CP_IDX = {'C': 0, 'P': 1}
    MARGIN_TYPE = ['ori_a', 'maint_a', 'sett_a', 'ori_b', 'maint_b', 'sett_b']
    MARGIN_TYPE_IDX = {'ori_a': 0, 'maint_a': 1, 'sett_a': 2, 'ori_b': 3, 'maint_b': 4, 'sett_b': 5}
    
    def __init__(self, option_folder):
        # Env parameter initial
        self.option_folder = option_folder
        self.done = True
        
    # 讀取台股交易日
    def load_trading_day(self):
        df = pd.read_excel(self.option_folder + 'Y9999.xlsx')
        start_date_row = df.loc[df['年月日'] == self.start_date]
        assert len(start_date_row) > 0, '起始日無交易'
        start_date_index = start_date_row.index[0]
        assert start_date_index >= self.history_steps, '交易日資料不足'
        head_date_index = start_date_index - self.history_steps
        end_date_index = start_date_index + self.steps
        self.taiex_open = df['開盤價(元)'].iloc[head_date_index:end_date_index].values.astype(int)
        self.taiex_close = df['收盤價(元)'].iloc[head_date_index:end_date_index].values.astype(int)
        self.trading_day = pd.DatetimeIndex(df['年月日'].iloc[head_date_index:end_date_index])
        
    def load_margin(self):
#        self.margin = pd.read_csv(self.option_folder + 'margin.csv') #保證金
#        self.margin['start'] = pd.to_datetime(self.margin['start'])
#        self.margin['end'] = pd.to_datetime(self.margin['end'])
        
        df = pd.read_csv(self.option_folder + 'margin.csv') #保證金
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])
        
        self.margin = pd.DataFrame(index = self.trading_day, columns = self.MARGIN_TYPE)
        for i, row in df.iterrows():
            for date in self.margin.loc[row['start']:row['end'] + pd.Timedelta('1 days')].index:
                self.margin.loc[date, self.MARGIN_TYPE] = row[self.MARGIN_TYPE]
        self.margin = self.margin.values
        
    def load_settlement_price(self):
        self.settlement_price = pd.read_csv(self.option_folder + 'settlement.csv')
        self.settlement_price['Date'] = pd.to_datetime(self.settlement_price['Date'])
        self.settlement_price = self.settlement_price.set_index('Date')
    
    def load_price(self):
        global txo
        
        txo_01_C = pd.read_csv(self.option_folder + 'txo_1_c.csv', index_col = 0)
        txo_01_P = pd.read_csv(self.option_folder + 'txo_1_p.csv', index_col = 0)
        txo_02_C = pd.read_csv(self.option_folder + 'txo_2_c.csv', index_col = 0)
        txo_02_P = pd.read_csv(self.option_folder + 'txo_2_p.csv', index_col = 0)
        txo = [[txo_01_C, txo_01_P], [txo_02_C, txo_02_P]]
        
        self.sp = np.unique(np.concatenate(
                [txo[m][n]['Strike Price'].values
                for m, n in product(range(2), range(2))], axis=None))
        self.sp_cnt = len(self.sp)
        self.sp_idx = {j: i for i, j in enumerate(self.sp)}
        
        total_steps = self.steps + self.history_steps
        open_prem = np.zeros((total_steps, 2, 2, self.sp_cnt))
        close_prem = np.zeros((total_steps, 2, 2, self.sp_cnt))
        
        for m, n in product(range(2), range(2)):
            df = txo[m][n]
            df['Date'] = pd.to_datetime(df['Date'])
            date_grouped = df.groupby('Date')
            for s in range(total_steps):
                for _, row in date_grouped.get_group(self.trading_day[s]).iterrows():
                    sp = self.sp_idx[row['Strike Price']]
                    open_prem[s, m, n, sp] = row['Open']
                    close_prem[s, m, n, sp] = row['Close']
        
        self.open = open_prem
        self.close = close_prem
    
    def reset(
            self, 
            cash,
            start_date, 
            steps,
            history_steps):

        self.cash = cash
        self.start_date = start_date
        self.steps = steps
        self.history_steps = history_steps
        
        self.load_trading_day()
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
        
    def __update_margin(self, date_index):
        self.margin_ori_a = self.margin[date_index, self.MARGIN_TYPE_IDX['ori_a']]
        self.margin_maint_a = self.margin[date_index, self.MARGIN_TYPE_IDX['maint_a']]
        self.margin_ori_b = self.margin[date_index, self.MARGIN_TYPE_IDX['ori_b']]
        self.margin_maint_b = self.margin[date_index, self.MARGIN_TYPE_IDX['maint_b']]
                
    def __update_margin_lvl(self, target_price):
        position_volume = np.abs(self.position)
        cond = self.position < 0
        margin_ori = np.zeros(self.position.shape, dtype=int)
        margin_maint = np.zeros(self.position.shape, dtype=int)
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m, n])[0]:    
                otm_value = max((self.sp[i] - target_price) * (-1) ** n, 0)
                margin_ori[m, n, i] = (
                        sum(self.position_queue[m][n][i]) * self.MULTIPLIER
                      + max(self.margin_ori_a - otm_value, self.margin_ori_b) * position_volume[m, n, i])
                margin_maint[m, n, i] = (
                        sum(self.position_queue[m][n][i]) * self.MULTIPLIER
                      + max(self.margin_maint_a - otm_value, self.margin_maint_b) * position_volume[m, n, i])
        self.margin_ori_lvl = np.sum(margin_ori[cond])
        self.margin_maint_lvl = np.sum(margin_maint[cond])
           
    def __new(self, order, cond, open_prem, target_price):
        # long option
        cond_long = cond & (order > 0)
        deal_long, cost = self.__long(order, cond_long, open_prem)

        # short option
        cond_short = cond & (order < 0)
        deal_short, premium = self.__short(order, cond_short, open_prem, target_price)
        
        return deal_long + deal_short, cost, premium
    
    def __long(self, order, cond, open_prem):
        deal_long = order.copy()
        deal_long[np.logical_not(cond)] = 0
        
        total_cost = int(np.sum(open_prem * order) * self.MULTIPLIER)
        if self.pool < total_cost + self.margin_ori_lvl:
            diff = total_cost + self.margin_ori_lvl - self.pool
            if diff > self.cash:
                tmp_cash = self.cash + self.pool - self.margin_ori_lvl
                for m, n in product(range(2), range(2)):
                    for i in np.where(cond[m, n])[0]:
                        unit_price = open_prem[m, n, i] * self.MULTIPLIER
                        max_volume = int(tmp_cash / unit_price)
                        if max_volume < deal_long[m, n, i]:
                            deal_long[m, n, i] = max_volume
                        tmp_cash -= unit_price * deal_long[m, n, i]
                total_cost = int(np.sum(open_prem * deal_long) * self.MULTIPLIER)
                diff = total_cost + self.margin_ori_lvl - self.pool
            # 入金
            self.pool += diff
            self.cash -= diff
        
        # 從保證金帳戶扣權利金
        self.pool -= total_cost
        
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m, n])[0]:
                self.position_queue[m][n][i].extend([open_prem[m, n, i]] * order[m, n, i])
        self.position += deal_long
        
        return deal_long, total_cost
    
    def __short(self, order, cond, open_prem, target_price):
        '''
        call價外值： MAXIMUM((履約價格-標的價格) ×契約乘數,0)
        put價外值： MAXIMUM((標的價格-履約價格)×契約乘數,0)
        保證金：權利金市值＋MAXIMUM (A值-價外值, B值)
        '''
        deal_short = order.copy()
        deal_short[np.logical_not(cond)] = 0
        volume = np.abs(deal_short)
        
        margin = np.zeros(order.shape, dtype=int)
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m, n])[0]:    
                otm_value = max((self.sp[i] - target_price) * (-1) ** n, 0)
                margin[m, n, i] = int(
                        open_prem[m, n, i] * self.MULTIPLIER 
                        + max(self.margin_ori_a - otm_value,
                              self.margin_ori_b))
        total_margin = np.sum(margin * volume)
        
        if self.pool < (total_margin + self.margin_ori_lvl):
            diff = total_margin + self.margin_ori_lvl - self.pool
            if diff > self.cash:
                tmp_cash = self.cash + self.pool - self.margin_ori_lvl
                for m, n in product(range(2), range(2)):
                    for i in np.where(cond[m, n])[0]:
                        max_volume = int(tmp_cash / margin[m, n, i])
                        if max_volume < volume[m, n, i]:
                            # 現金不足，計算最大可買張數
                            volume[m, n, i] = max_volume
                        tmp_cash -= margin[m, n, i] * volume[m, n, i]
                deal_short = volume * -1
                total_margin = np.sum(margin * volume)
                diff = total_margin + self.margin_ori_lvl - self.pool
            self.pool += diff
            self.cash -= diff
        
        # 收權利金
        premium = int(np.sum(open_prem * volume) * self.MULTIPLIER)
        self.pool += premium
        
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m, n])[0]:
                self.position_queue[m][n][i].extend([open_prem[m, n, i]] * volume[m, n, i])
        self.position += deal_short
        self.margin_ori_lvl += total_margin
    
        return deal_short, premium
    
    def __close(self, order, cond, open_prem, target_price):
        deal_close = order.copy()
        deal_close[np.logical_not(cond)] = 0
        
        volume = np.abs(deal_close)
        position_volume = np.abs(self.position)
        
        # 平倉量超出庫存
        cond_over_sell = cond & (volume > position_volume)
        deal_close[cond_over_sell] = self.position[cond_over_sell] * -1
        volume = np.abs(deal_close)
        
        # 平倉權利金
        close_point = open_prem * deal_close
        
        # 庫存點位
        position_point = np.zeros(self.position.shape)
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m][n])[0]:
                for j in range(volume[m, n, i]):
                    assert len(self.position_queue[m][n][i]) > 0
                    position_point[m, n, i] += self.position_queue[m][n][i].popleft()
        position_point *= np.sign(self.position)
        profit = int(np.sum(close_point + position_point)) * -1 * self.MULTIPLIER
        
        self.position += deal_close
        self.pool += profit
        
        # 調整保證金水位
        self.__update_margin_lvl(target_price)
        
        return deal_close, profit
    
    def __settlement(self, date_index):
        # 履約價
        strike_point = self.position[0] * self.sp
        
        # 結算價
        final_price = self.settlement_price.loc[self.trading_day[date_index], 'Price']
        
        # 結算點位
        settlement_point = self.position[0] * final_price
        
        profit = np.sum(np.maximum(settlement_point - strike_point, 0).T * [1, -1]) * self.MULTIPLIER
        self.pool += profit
        
        self.position[0] = 0
        position_point = np.zeros(self.position.shape)
        for m, n, i in product(range(2), range(2), range(self.sp_cnt)):
            position_point[m, n, i] = sum(self.position_queue[m][n][i]) * np.sign(self.position[m, n, i])
        close_point = (self.close[date_index] * self.position).astype(int)
        unrealized = int(np.sum(close_point - position_point)) * self.MULTIPLIER
        
        # 轉倉
        self.position[0] = self.position[1]
        self.position[1] = 0
        
        for n, o in product(range(2), range(self.sp_cnt)): 
            self.position_queue[0][n][o] = self.position_queue[1][n][o].copy()
            self.position_queue[1][n][o].clear()

        return profit, unrealized
    
    def step(self, action):
        date_index = self.cnt + self.history_steps
        self.__update_margin(date_index)

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
                        self.open[date_index],
                        self.taiex_open[date_index])
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
                                                self.open[date_index], 
                                                self.taiex_open[date_index])
        order -= deal_close
        profit += profit_close
        
        # 建倉 / 新倉
        cond_new = np.logical_not(cond_close)
        deal_new, cost, premium = self.__new(order, 
                                             cond_new, 
                                             self.open[date_index],
                                             self.taiex_open[date_index])
        
        order_deal = deal_close + deal_new + deal_liq
        
        # 結算
        unrealized = 0
        if self.trading_day[date_index] in self.settlement_price.index:
            profit_sett, unrealized = self.__settlement(date_index)
            profit += profit_sett
        else:
            # 未實現損益
            position_point = np.zeros(self.position.shape)
            for m, n, i in product(range(2), range(2), range(self.sp_cnt)):
                position_point[m, n, i] = sum(self.position_queue[m][n][i]) * np.sign(self.position[m, n, i])
            close_point = self.close[date_index] * self.position
            unrealized = int(np.sum(close_point - position_point)) * self.MULTIPLIER
        
        # 檢查保證金水位
        self.__update_margin_lvl(self.taiex_close[date_index])
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
    option_folder = './option_data/'
    
    start_date = '2016-01-19'
    period = 4
    action = list([
            [['TXO01', 'C', 7900, 1], ['TXO02', 'C', 7900, -2], ['TXO01', 'P', 7700, -1]],
            [['TXO02', 'P', 8700, 2], ['TXO01', 'C', 7900, 2], ['TXO01', 'P', 8500, 1]],
            [['TXO01', 'C', 9200, 1],['TXO02', 'C', 9200, -1]],
            [['TXO01', 'P', 7700, -1]]
            ])   
    
    cash = 1e+6
    env = Env(option_folder)  
    env.reset(cash, start_date, period, 0)
    total_time = 0
    for i in range(period):
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