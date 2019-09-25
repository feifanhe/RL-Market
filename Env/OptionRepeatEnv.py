# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:36:33 2019

@author: Fei-fan He
"""

import numpy as np
from collections import deque
from itertools import product
import OptionEnv

class Env(OptionEnv.Env):
    def __init__(self, option_folder):
        super().__init__(option_folder)
        
    def reset(self,
              repeat,
              cash,
              start_date,
              steps,
              history_steps):
        super().reset(
                cash,
                start_date,
                steps,
                history_steps)
        self.repeat = repeat
        self.cash = np.array([cash] * self.repeat, dtype = int)
        self.pool = np.zeros(self.repeat, dtype = int)
        self.margin_ori_lvl = np.zeros(self.repeat, dtype = int)
        self.margin_maint_lvl = np.zeros(self.repeat, dtype = int)
        self.margin_call = np.zeros(self.repeat, dtype = int)
        self.position = np.zeros((self.repeat, 2, 2, self.sp_cnt), dtype = int)
        self.position_queue = [[[[
                deque([])
                for _ in range(self.sp_cnt)]
                for _ in range(2)]
                for _ in range(2)]
                for _ in range(self.repeat)]
        self.open = np.tile(self.open, (self.repeat, 1, 1, 1, 1))
        self.close = np.tile(self.open, (self.repeat, 1, 1, 1, 1))
        
    def __parse_order(self, actions):
        order = np.zeros(self.position.shape, dtype = int)
        for i, action in enumerate(actions):
            for near_month, cp, sp, volume in action:
                order[i,
                      self.CONTRACT_IDX[near_month], 
                      self.CP_IDX[cp],
                      self.sp_idx[sp]] = volume
        return order
    
    def __update_margin_lvl(self, target_price):
        position_volume = np.abs(self.position)
        cond = self.position < 0
        margin_ori = np.zeros(self.position.shape, dtype = int)
        margin_maint = np.zeros(self.position.shape, dtype = int)
        for r, m, n in product(range(self.repeat), range(2), range(2)):
            for i in np.where(cond[r, m, n])[0]:    
                otm_value = max((self.sp[i] - target_price) * (-1) ** n, 0)
                margin_ori[r, m, n, i] = (
                        sum(self.position_queue[r][m][n][i]) * self.MULTIPLIER
                      + max(self.margin_ori_a - otm_value, self.margin_ori_b) * position_volume[r, m, n, i])
                margin_maint[r, m, n, i] = (
                        sum(self.position_queue[r][m][n][i]) * self.MULTIPLIER
                      + max(self.margin_maint_a - otm_value, self.margin_maint_b) * position_volume[r, m, n, i])
        self.margin_ori_lvl = np.sum(margin_ori, axis = (1, 2, 3))
        self.margin_maint_lvl = np.sum(margin_maint, axis = (1, 2, 3))
        
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
        for r, m, n in product(range(self.repeat), range(2), range(2)):
            for i in np.where(cond[r][m][n])[0]:
                for _ in range(volume[r, m, n, i]):
                    assert len(self.position_queue[r][m][n][i]) > 0
                    position_point[r, m, n, i] += self.position_queue[r][m][n][i].popleft()
        position_point *= np.sign(self.position)
        profit = np.sum(close_point + position_point, axis = (1, 2, 3)).astype(int) * -1 * self.MULTIPLIER
        
        self.position += deal_close
        self.pool += profit
        
        # 調整保證金水位
        self.__update_margin_lvl(target_price)
        
        return deal_close, profit
    
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
        
        total_cost = (np.sum(open_prem * order, axis = (1, 2, 3)) * self.MULTIPLIER).astype(int)
        
        cond_cash_not_enough = self.cash < total_cost + self.margin_ori_lvl - self.pool
        tmp_cash = self.cash + self.pool - self.margin_ori_lvl
        for r in np.where(cond_cash_not_enough)[0]:
            for m, n in product(range(2), range(2)):
                for i in np.where(cond[r, m, n])[0]:
                    unit_price = open_prem[r, m, n, i] * self.MULTIPLIER
                    max_volume = int(tmp_cash[r] / unit_price)
                    if max_volume < deal_long[r, m, n, i]:
                        deal_long[r, m, n, i] = max_volume
                    tmp_cash[r] -= unit_price * deal_long[r, m, n, i]
        total_cost = (np.sum(open_prem * deal_long, axis = (1, 2, 3)) * self.MULTIPLIER).astype(int)
        diff = total_cost + self.margin_ori_lvl - self.pool
        diff[diff <= 0] = 0
        self.pool += diff
        self.cash -= diff
        
        self.pool -= total_cost
        
        for r, m, n in product(range(self.repeat), range(2), range(2)):
            for i in np.where(cond[r, m, n])[0]:
                self.position_queue[r][m][n][i].extend([open_prem[r, m, n, i]] * order[r, m, n, i])
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
        for r in range(self.repeat):
            for m, n in product(range(2), range(2)):
                for i in np.where(cond[r, m, n])[0]:    
                    otm_value = max((self.sp[i] - target_price) * (-1) ** n, 0)
                    margin[r, m, n, i] = int(
                            open_prem[r, m, n, i] * self.MULTIPLIER 
                            + max(self.margin_ori_a - otm_value,
                                  self.margin_ori_b))
        total_margin = np.sum(margin * volume, axis = (1, 2, 3))
        
        cond_cash_not_enough = self.cash < total_margin + self.margin_ori_lvl - self.pool
        tmp_cash = self.cash + self.pool - self.margin_ori_lvl
        for r in np.where(cond_cash_not_enough)[0]:
            for m, n in product(range(2), range(2)):
                for i in np.where(cond[r, m, n])[0]:
                    max_volume = int(tmp_cash[r] / margin[r, m, n, i])
                    if max_volume < volume[r, m, n, i]:
                        # 現金不足，計算最大可買張數
                        volume[r, m, n, i] = max_volume
                    tmp_cash -= margin[r, m, n, i] * volume[r, m, n, i]
        deal_short = volume * -1
        total_margin = np.sum(margin * volume, axis = (1, 2, 3))
        diff = total_margin + self.margin_ori_lvl - self.pool
        diff[diff <= 0] = 0
        self.pool += diff
        self.cash -= diff
        
        premium = int(np.sum(open_prem * volume, axis = (1, 2, 3)) * self.MULTIPLIER)
        self.pool += premium
        
        for r in range(self.repeat):
            for m, n in product(range(2), range(2)):
                for i in np.where(cond[r, m, n])[0]:
                    self.position_queue[r][m][n][i].extend([open_prem[r, m, n, i]] * volume[r, m, n, i])
        self.position += deal_short
        self.margin_ori_lvl += total_margin
    
        return deal_short, premium
    
    def __settlement(self, date_index):
        # 履約價
        strike_point = self.position[:, 0] * self.sp
        
        # 結算價
        final_price = self.settlement_price.loc[self.trading_day[date_index], 'Price']
        
        # 結算點位
        settlement_point = self.position[:, 0] * final_price
        
        diff = np.maximum(settlement_point - strike_point, 0)
        diff = diff[:, 0] - diff[:, 1]
        print(diff)
        print(diff.shape)

        profit = np.sum(diff, axis = 1) * self.MULTIPLIER
        self.pool += profit
        
        self.position[:, 0] = 0
        position_point = np.zeros(self.position.shape)
        for r, m, n, i in product(range(self.repeat), range(2), range(2), range(self.sp_cnt)):
            position_point[r, m, n, i] = sum(self.position_queue[r][m][n][i]) * np.sign(self.position[r, m, n, i])
        close_point = (self.close[:, date_index] * self.position).astype(int)
        unrealized = (np.sum(close_point - position_point, axis = (1, 2, 3)) * self.MULTIPLIER).astype(int)
        
        # 轉倉
        self.position[:, 0] = self.position[:, 1]
        self.position[:, 1] = 0
        
        for r, n, o in product(range(self.repeat), range(2), range(self.sp_cnt)): 
            self.position_queue[r][0][n][o] = self.position_queue[r][1][n][o].copy()
            self.position_queue[r][1][n][o].clear()

        return profit, unrealized
    
    def step(self, actions):
        date_index = self.cnt + self.history_steps
        self.__update_margin(date_index)

        profit = np.zeros(self.repeat, dtype = int)
        
        # 追繳保證金，先檢查強制平倉
        cond_cash_not_enough = self.cash < self.margin_call
        cond_liq = self.position < 0
        cond_liq[np.logical_not(cond_cash_not_enough)] = False
        deal_liq, profit_liq = self.__close(
                self.position * -1,
                cond_liq,
                self.open[:, date_index],
                self.taiex_open[date_index])
        self.margin_call[cond_cash_not_enough] = 0
        # 扣保證金
        self.cash -= self.margin_call
        self.pool += self.margin_call
        self.margin_call[:] = 0
        
        # 委託
        order = self.__parse_order(actions)
        order_original = order.copy()
        
        # 平倉
        cond_close = order * self.position < 0
        deal_close, profit_close = self.__close(order,
                                                cond_close,
                                                self.open[:, date_index],
                                                self.taiex_close[date_index])
        order -= deal_close
        profit += profit_close
        
        # 建倉 / 新倉
        cond_new = np.logical_not(cond_close)
        deal_new, cost, premium = self.__new(order, 
                                             cond_new, 
                                             self.open[:, date_index],
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
            for r, m, n, i in product(range(self.repeat), range(2), range(2), range(self.sp_cnt)):
                position_point[r, m, n, i] = sum(self.position_queue[r][m][n][i]) * np.sign(self.position[r, m, n, i])
            close_point = self.close[:, date_index] * self.position
            unrealized = (np.sum(close_point - position_point, axis = (1, 2, 3)) * self.MULTIPLIER).astype(int)
        
        # 檢查保證金水位
        self.__update_margin_lvl(self.taiex_close[date_index])
        # 低於維持保證金，追繳至原始保證金
        cond_margin_call = self.pool < self.margin_maint_lvl
        diff = self.margin_ori_lvl - self.pool
        self.margin_call[cond_margin_call] = diff
        
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
    option_folder = './option_data/'
    
    repeat = 1
    start_date = '2016/01/19'
    period = 4
    action = list([
            [[['TXO01', 'C', 7900, 1], ['TXO02', 'C', 7900, -2], ['TXO01', 'P', 7700, -1]]],
            [[['TXO02', 'P', 8700, 2], ['TXO01', 'C', 7900, 2], ['TXO01', 'P', 8500, 1]]],
            [[['TXO01', 'C', 9200, 1],['TXO02', 'C', 9200, -1]]],
            [[['TXO01', 'P', 7700, -1]]]
            ])   
    
    cash = 1e+6
    env = Env(option_folder)  
    env.reset(repeat, cash, start_date, period, 0)
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