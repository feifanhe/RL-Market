# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:22:17 2019

@author: feifanhe
"""
import pandas as pd
import numpy as np
from collections import deque
from itertools import product

class Env:
    # Constants
    TAX_RATE = 0.001 #交易稅
    TAX_RATE_SETT = 0.00002 #交易稅(結算)
    FEE = 15 #手續費
    MULTIPLIER = 50
    NEAR_MONTH_IDX = {'TXO01': 0, 'TXO02': 1}
    CP_IDX = {'C': 0, 'P': 1}
           
        
    def __init__(self, option_folder):
        # Env parameter initial
        self.option_folder = option_folder
    
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
        self.margin = pd.read_csv(self.option_folder + 'margin.csv') #保證金
        self.margin['start'] = pd.to_datetime(self.margin['start'])
        self.margin['end'] = pd.to_datetime(self.margin['end'])
        
    def load_settlement_price(self):
        self.settlement_price = pd.read_csv(self.option_folder + 'settlement.csv')
        self.settlement_price['Date'] = pd.to_datetime(self.settlement_price['Date'])
        self.settlement_price = self.settlement_price.set_index('Date')
    
    def load_price(self):
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
    
    def __update_margin(self, date_index):
        for index, row in self.margin.iterrows():
            if row['start'] <= self.trading_day[date_index] <= row['end']:
                self.margin_ori_a = row['ori_a']
                self.margin_maint_a = row['maint_a']
                self.margin_ori_b = row['ori_b']
                self.margin_maint_b = row['maint_b']
                
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
        deal_new = order.copy()
        
        # long option
        cond_long = cond & (deal_new > 0)
        cost = self.__long(deal_new, cond_long, open_prem)

        # short option
        cond_short = cond & (deal_new < 0)
        premium = self.__short(deal_new, cond_short, open_prem, target_price)
        
        return deal_new, cost, premium
    
    def __long(self, order, cond, open_prem):
        total_cost = int(np.sum(open_prem[cond] * order[cond]) * self.MULTIPLIER)
        if self.pool < total_cost + self.margin_ori_lvl:
            diff = total_cost + self.margin_ori_lvl - self.pool
            if diff > self.cash:
                tmp_cash = self.cash + self.pool - self.margin_ori_lvl
                for m, n in product(range(2), range(2)):
                    for i in np.where(cond[m, n])[0]:
                        unit_price = open_prem[m, n, i] * self.MULTIPLIER
                        max_volume = int(tmp_cash / unit_price)
                        if max_volume < order[m, n, i]:
                            order[m, n, i] = max_volume
                        tmp_cash -= unit_price * order[m, n, i]
                total_cost = int(np.sum(open_prem[cond] * order[cond]) * self.MULTIPLIER)
                diff = total_cost + self.margin_ori_lvl - self.pool
            # 入金
            self.pool += diff
            self.cash -= diff
        
        # 從保證金帳戶扣權利金
        self.pool -= total_cost
        
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m, n])[0]:
                self.position_queue[m][n][i].extend([open_prem[m, n, i]] * order[m, n, i])
        self.position[cond] += order[cond]
        
        return total_cost
    
    def __short(self, order, cond, open_prem, target_price):
        '''
        call價外值： MAXIMUM((履約價格-標的價格) ×契約乘數,0)
        put價外值： MAXIMUM((標的價格-履約價格)×契約乘數,0)
        保證金：權利金市值＋MAXIMUM (A值-價外值, B值)
        '''
        volume = np.abs(order)
        margin = np.zeros(order.shape, dtype=int)
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m, n])[0]:    
                otm_value = max((self.sp[i] - target_price) * (-1) ** n, 0)
                margin[m, n, i] = int(
                        open_prem[m, n, i] * self.MULTIPLIER 
                        + max(self.margin_ori_a - otm_value,
                              self.margin_ori_b))
        total_margin = np.sum(margin[cond] * volume[cond])
        
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
                order[cond] = volume[cond] * -1
                total_margin = np.sum((margin * volume)[cond])
                diff = total_margin + self.margin_ori_lvl - self.pool
            self.pool += diff
            self.cash -= diff
        
        # 收權利金
        premium = int(np.sum(open_prem[cond] * volume[cond]) * self.MULTIPLIER)
        self.pool += premium
        
        for m, n in product(range(2), range(2)):
            for i in np.where(cond[m, n])[0]:
                self.position_queue[m][n][i].extend([open_prem[m, n, i]] * volume[m, n, i])
        self.position[cond] += order[cond]
        self.margin_ori_lvl += total_margin
    
        return premium
    
    def __close(self, order, cond, open_prem, target_price):
        deal_close = order.copy()
        deal_close[np.logical_not(cond)] = 0
        
        volume = np.abs(deal_close)
        position_volume = np.abs(self.position)
        
        # 平倉量超出庫存
        cond_over_sell = cond & (volume > position_volume)
        deal_close[cond_over_sell] = position_volume[cond_over_sell] * -1
        volume[cond_over_sell] = np.abs(deal_close[cond_over_sell])
        
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
        
        return profit, deal_close
    
    def __settlement(self, date_index):
        # 履約價
        strike_point = self.position[0] * self.sp
        
        # 結算價
        final_price = self.settlement_price.loc[self.trading_day[date_index], 'Price']
        
        # 結算點位
        settlement_point = self.position[0] * final_price
        
        profit = np.sum(np.maximum(settlement_point - strike_point, 0).T * [1, -1]) * self.MULTIPLIER
        self.pool += profit
        
        # 轉倉
        self.position[0] = self.position[1]
        self.position[1] = 0
        
        for n, o in product(range(2), range(self.sp_cnt)): 
            self.position_queue[0][n][o] = self.position_queue[1][n][o].copy()
            self.position_queue[1][n][o].clear()

        return profit
    
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
                profit_liq, deal_liq = self.__close(self.position * -1, cond_liq, self.open[date_index], self.taiex_open[date_index])
                profit += profit_liq
            else:
               self.cash -= self.margin_call
               self.pool += self.margin_call
               self.margin_call = 0
        
        # 委託
        order = np.zeros(self.position.shape, dtype=int)
        for near_month, cp, sp, volume in action:
            order[self.NEAR_MONTH_IDX[near_month], 
                  self.CP_IDX[cp],
                  self.sp_idx[sp]] = volume
        order_original = order.copy()
        
        # 先平倉
        cond_close = (order * self.position) < 0
        profit_close, deal_close = self.__close(order, 
                                                cond_close, 
                                                self.open[date_index], 
                                                self.taiex_open[date_index])
        profit += profit_close
        order -= deal_close
        
        # 建倉 / 新倉
        cond_new = (order * self.position) >= 0
        deal_new, cost, premium = self.__new(order, 
                                             cond_new, 
                                             self.open[date_index],
                                             self.taiex_open[date_index])
        order_deal = deal_close + deal_new + deal_liq
        
        # 結算
        if self.trading_day[date_index] in self.settlement_price.index:
            profit += self.__settlement(date_index)
            
        # 未實現損益 = 
        position_point = self.sp * self.position
        close_point = self.taiex_close[date_index] * self.position 
        diff = np.sum(np.maximum(close_point - position_point, 0).T * [1, -1]) * self.MULTIPLIER
        unrealized = premium - cost + diff
        
        # 檢查保證金水位
        self.__update_margin_lvl(self.taiex_close[date_index])
        if self.pool < self.margin_maint_lvl:
            # 低於維持保證金，追繳至原始保證金
            self.margin_call = self.margin_ori_lvl - self.pool
        
        self.cnt += 1
        
        return (self.cash, 
                self.pool,
                cost,       # 權利金支出
                premium,    # 權利金收入
                unrealized, 
                profit, 
                self.position, 
                order_original, 
                order_deal, 
                self.margin_call
                )
        
#%%
if __name__ == '__main__':
    option_folder = './option_data/'
    
    start_date = '2016/01/19'
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
    
    for i in range(period):
        print(f'[Step {i+1}]')
        print(action[i])
        cash, pool, cost, premium, unrealized, profit, position, order, deal, margin_call = env.step(action[i])
        
        print('Position:\n', position)
        print('Cost\tPremium\tProfit\tUnrl\tMargin Call')
        print(f'{cost}\t{premium}\t{profit}\t{unrealized}\t{margin_call}')
        print('Cash remains:', cash)
        print('Pool remains:', pool)
        print()
