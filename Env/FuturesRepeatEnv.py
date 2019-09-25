# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:45:51 2019

@author: Fei-fan He
"""

#%%
import numpy as np
from collections import deque
import FuturesEnv

#%%
class Env(FuturesEnv.Env):
    def __init__(
            self,
            futures_folder):
        super().__init__(futures_folder)
        
    def reset(
            self,
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
        self.margin_ori_level = np.zeros(self.repeat, dtype = int)
        self.margin_call = np.zeros(self.repeat, dtype = int)
        
        # reset position
        self.position_shape = (self.repeat, self.CONTRACT_COUNT)
        self.position = np.zeros(self.position_shape, dtype = int)
        self.position_queue = [[deque([]) for _ in range(self.CONTRACT_COUNT)] for _ in range(self.repeat)]
        
        self.open = np.tile(self.open, (self.repeat, 1, 1))
        self.close = np.tile(self.close, (self.repeat, 1, 1))
        self.margin_ori = np.tile(self.margin_ori, (self.repeat, 1, 1))
        self.margin_maint = np.tile(self.margin_maint, (self.repeat, 1, 1))
        
        self.done = False
        
    def __parse_order(self, actions):
        # 委託單
        order = np.zeros(self.position_shape, dtype = int)
        for i, action in enumerate(actions):
            for symbol, volume in action:
                order[i, self.CONTRACT_IDX[symbol]] = volume
        return order
    
    def __close(self, order, cond, open_price, margin_ori):
        deal_close = order.copy()
        deal_close[np.logical_not(cond)] = 0
        volume = np.abs(deal_close)
        position_volume = np.abs(self.position)

        # 平倉量超出庫存
        cond_over_sell = cond & (volume > position_volume)
        deal_close[cond_over_sell] = self.position[cond_over_sell] * -1
        volume = np.abs(deal_close)
        
        # 平倉點位
        close_point = open_price * deal_close
        
        # 庫存點位
        position_point = np.zeros(self.position_shape, dtype = int)
        for i in range(self.repeat):
            for j in np.where(cond[i])[0]:
                for k in range(volume[i, j]):
                    assert len(self.position_queue[i][j]) > 0
                    position_point[i, j] += int(self.position_queue[i][j].popleft())
        position_point *= np.sign(self.position)
        
        profit = np.sum((close_point + position_point) * -1 * self.CONTRACT_SIZE, axis = 1)
        self.position += deal_close
        
        self.margin_ori_level = self.margin_ori_level - np.sum(margin_ori * volume, axis = 1)
        self.pool = self.pool + profit
        
        return profit, deal_close
        
    def __new(self, order, cond, open_price, margin_ori):
        deal_new = order.copy()
        deal_new[np.logical_not(cond)] = 0
        volume = np.abs(deal_new)
        margin = np.sum(margin_ori * volume, axis = 1)
        
        cond_cash_not_enough = margin + self.margin_ori_level - self.pool < self.cash
        for i in np.where(cond_cash_not_enough)[0]:
            tmp_cash = self.cash[i] + self.pool[i] - self.margin_ori_level
            for j in np.where(cond[i])[0]:
                if (tmp_cash / margin_ori[i, j]) < volume[i, j]:
                    # 現金不足，計算最大可買張數
                    volume[i, j] = int(tmp_cash / margin_ori[i, j])
                tmp_cash -= margin_ori[i, j] * volume[i, j]
            deal_new[i] = np.sign(deal_new[i]) * volume[i]
            margin[i] = np.sum(margin_ori[i] * volume[i])
        
        diff = margin + self.margin_ori_level - self.pool
        diff[diff <= 0] = 0
        self.pool = self.pool + diff
        self.cash = self.cash - diff
        
        for i in range(self.repeat):
            for j in np.where(cond[i])[0]:
                self.position_queue[i][j].extend([open_price[i][j]] * volume[i][j])
                
        self.position += deal_new
        self.margin_ori_level += margin

        return deal_new
    
    def __settlement(self, date_index):
        # 結算價
        final_price = self.settlement_price.loc[self.trading_day[date_index], 'Price']
        
        # 庫存點位
        position_point = np.zeros(self.position_shape, dtype = int)
        for i in range(self.repeat):
            for j in [0, 2]:
                position_point[i, j] = sum(self.position_queue[i][j]) * np.sign(self.position[i, j])
        
        # 結算點位
        settlement_point = self.position * final_price
        settlement_point[:, [1, 3]] = 0
        profit = np.sum((settlement_point - position_point) * self.CONTRACT_SIZE, axis = 1)
        self.pool += profit
        
        # 轉倉
        self.position[:, [0, 2]] = self.position[:, [1, 3]]
        self.position[:, [1, 3]] = 0
        for i in range(self.repeat):
            self.position_queue[i][0] = self.position_queue[i][1].copy()
            self.position_queue[i][2] = self.position_queue[i][3].copy()
            self.position_queue[i][1].clear()
            self.position_queue[i][3].clear()
        
        # 調整保證金水位
        self.margin_ori_level = np.sum(self.margin_ori[:, date_index] * np.abs(self.position), axis = 1)
        
        return profit
        
    def step(self, actions):
        date_index = self.cnt + self.history_steps
        profit = np.zeros(self.repeat, dtype = int)
        
        # 強制平倉
        cond_cash_not_enough = self.cash < self.margin_call
        cond_liq = self.position != 0
        cond_liq[np.logical_not(cond_cash_not_enough)] = False
        profit_liq, deal_liq = self.__close(
                self.position * -1,
                cond_liq,
                self.open[:, date_index],
                self.margin_ori[:, date_index])
        self.margin_call[cond_cash_not_enough] = 0
        
        # 追繳保證金
        self.cash -= self.margin_call
        self.pool += self.margin_call
        
        self.margin_call[:] = 0
        
        # 委託單
        order = self.__parse_order(actions)
        order_original = order.copy()
        
        # 先平倉
        cond_close = (order * self.position) < 0
        profit_close, deal_close = self.__close(order, cond_close, self.open[:, date_index], self.margin_ori[:, date_index])
        profit = profit + profit_close
        order = order - deal_close
        
        # 建倉 / 新倉
        cond_new = (order * self.position) >= 0
        deal_new = self.__new(order, cond_new, self.open[:, date_index], self.margin_ori[:, date_index])
        
        order_deal = deal_liq + deal_close + deal_new
        
        # 結算
        if self.trading_day[date_index] in self.settlement_price.index:
            profit = profit + self.__settlement(date_index)
            
        # 庫存點位
        position_point = np.zeros(self.position_shape)
        for i in range(self.repeat):
            for j in range(self.CONTRACT_COUNT):
                position_point[i, j] = sum(self.position_queue[i][j])
        position_point *= np.sign(self.position)
        
        # average cost
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_cost = np.nan_to_num(position_point / self.position)
        
        # 未實現損益
        unrealized = np.sum((self.close[:, date_index] * self.position - position_point) * self.CONTRACT_SIZE, axis = 1)
        
        # 檢查保證金水位
        margin_maint_level = np.sum(self.margin_maint[:, date_index] * np.abs(self.position), axis = 1)
        cond_margin_call = self.pool + unrealized < margin_maint_level
        self.margin_call = self.margin_ori_level - (self.pool + unrealized)
        self.margin_call[np.logical_not(cond_margin_call)] = 0
        
        self.cnt += 1
        if self.cnt == self.steps:
            self.done = True
        
        return self.cash, self.pool, unrealized, profit, self.position, avg_cost, order_original, order_deal, self.margin_call

#%%
if __name__ == '__main__':
    
    import time
    
    futures_folder = './futures_data/'
    env = Env(futures_folder)
    
    repeat = 1
    cash = int(1e+6)
    start_date = '2016/01/19'
    steps = 5
    history_steps = 10
    
    env.reset(repeat,
              cash,
              start_date, 
              steps,
              history_steps)
    
    action = list([
            [[['TX01',-1],['TX02',-1]]],
            [[['TX01',-1],['TX02',1]]],
            [[['TX01',3],['TX02',3]]],
            [[]],
            [[['TX01',-3],['TX02',-3]]],
            ])
    total_time=0
    for i in range(steps):
        
        print(f'[step {i+1}]')
        start_time = time.time()
        cash, pool, unrealized, profit, position, avg_cost, order, deal, margin_call = env.step(action[i])
        end_time = time.time()
        print('Order:\t', order)
        print('Deal:\t', deal)
        print('Position:\t', position)
        print('Avg. cost:\t', avg_cost)
        print('Profit \t Unrealized \t Margin Call')
        print(profit, '\t', unrealized, '\t', margin_call)
        print('Cash remains:', cash)
        print('Pool remains:', pool)
        step_time = (end_time - start_time) * 1000
        total_time += step_time
        print(f'[Time: {step_time}ms]')
        print()
    print(f'[Total time: {total_time}ms]')
