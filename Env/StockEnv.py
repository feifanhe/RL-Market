# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:50:19 2019

@author: Kuo
"""
#%%
import copy
import pandas as pd
import numpy as np
from collections import deque

#%%
class Env():
    def __init__(self,
                  stock_folder, #股票所在資料夾
                  ):

        # Env parameter initial
        self.stock_folder = stock_folder
        
    def reset(
            self,
            cash, #初始化資金
            start_date, #交易起始日
            steps, #交易天數
            history_steps, #窺視歷史價格天數
            stock_targets, #要進行資產配置的股票
            ):
        
        self.cnt = 0
        self.cash = int(cash)
        self.start_date = start_date
        self.steps = steps
        self.history_steps = history_steps
        
        self.enable_fee = False
        
        self.stock_targets = stock_targets
        self.stock_targets_count = len(stock_targets)
        
        self.position = np.zeros(self.stock_targets_count)
        self.avr_cost = np.zeros(self.stock_targets_count)
        self.cost_queue = [deque([]) for _ in range(self.stock_targets_count)]
        
        # 建立台股交易日表格
        taiwan_index = pd.read_excel(self.stock_folder + 'Y9999.xlsx')
        start_day_row = taiwan_index.loc[taiwan_index['年月日'] == self.start_date]
        assert len(start_day_row) > 0, '起始日無交易'
        start_day_index = start_day_row.index[0]
        assert start_day_index >= self.history_steps, '交易日資料不足'
        
        self.trading_day = taiwan_index['年月日'].iloc[start_day_index - self.history_steps:start_day_index + self.steps].values
        
        # 讀取除息日
        self.dividend = pd.read_csv(self.stock_folder + 'dividend.csv')
        self.dividend = self.dividend[self.stock_targets].iloc[start_day_index - self.history_steps:start_day_index + self.steps].values
        
        # 讀取個股資料
        self.open = np.empty(self.stock_targets_count, dtype = object)
        self.close = np.empty(self.stock_targets_count, dtype = object)
        for i in range(self.stock_targets_count):
            tmp = pd.read_excel(
                    self.stock_folder +
                    self.stock_targets[i] + '/' + # 股號
                    self.start_date.split('/')[0] + # 年
                    '.xlsx', index_col = 0).reset_index(inplace=False)
            del tmp['index']
            idx = tmp.loc[tmp['年月日'] == self.start_date].index[0]
            assert start_day_index >= self.history_steps, '%d 缺少歷史交易資料' % self.stock_targets[i]
            tmp_trading_day = tmp['年月日'].iloc[idx-self.history_steps:idx+self.steps].values
            
            # TODO: 如果期間有缺失交易資料，回傳錯誤
            assert len(set(self.trading_day) - set(tmp_trading_day)) == 0,'%d 缺少交易資料' % self.stock_targets[i]
            
            self.open[i] = tmp.iloc[idx-self.history_steps:idx+self.steps, 1].values[None]
            self.close[i] = tmp.iloc[idx-self.history_steps:idx+self.steps, 4].values[None]
        
        self.open = np.concatenate(self.open).T
        self.close = np.concatenate(self.close).T
        self.stock_targets_idx = {j:i for i,j in enumerate(self.stock_targets)}
        
        return (self.close[self.cnt:self.cnt+self.history_steps], # 歷史收盤價
                self.cash, # 剩餘現金
                0, # 未實現資產市值
                0, # 損益
                self.avr_cost, # 平均成本
                np.zeros(self.stock_targets_count), # 原交易單
                np.zeros(self.stock_targets_count), # 實際交易單
                self.position # 持有部位
               )
        
    def step(self, action):
        time_index = self.cnt + self.history_steps
        print(self.trading_day[time_index])
        print(action)
        order = np.zeros(self.stock_targets_count, dtype=int)
        for code, amount in action:
            order[self.stock_targets_idx[code]] = amount

        order_result = copy.deepcopy(order)

        # 賣超過手中持有部位，action強制轉換成可賣出的上限
        cond_over_sell = ((self.position + order_result) < 0)
        order_result[cond_over_sell] = -1 * self.position[cond_over_sell]

        # 賣出
        cond_sell = (order_result < 0)  
        
        # 賣出時會計算 profit
        income = 0
        for i in np.where(cond_sell)[0]:
            i_income = int(self.open[time_index][i] * -1 * order_result[i] * 1000)
            i_income -= self.fee(i_income)
            income += i_income
        income -= int(income * 0.003)

        # 根據買進成本計算 cost
        cost = 0
        for i in np.where(cond_sell)[0]:
            i_cost = 0
            for j in range(order_result[i], 0):
                assert len(self.cost_queue[i]) > 0
                i_cost += int(self.cost_queue[i].popleft() * 1000)
            i_cost += self.fee(i_cost)
            cost += i_cost
            
        profit = income - cost
        self.cash += int(income)
        
        # 修正持有部位和平均成本
        self.position[cond_sell] += order_result[cond_sell]
        self.avr_cost[((self.position)==0)] = 0
        
        # 買入
        cond_buy = (order_result > 0) 
        
        # 檢查現金是否足夠
        cost = np.sum(self.open[time_index][cond_buy] * order_result[cond_buy] * 1000)
        cost += self.fee(cost)
        if self.cash < cost:
            # 修改 order_result[cond_buy]
            tmp_cash = self.cash
            for i in np.where(cond_buy)[0]:
                i_cost = self.open[time_index][i] * 1000
                i_cost += self.fee(i_cost)
                if (tmp_cash / i_cost) < order_result[i]:
                    # 現金不足，計算最大可買張數
                    order_result[i] = int(tmp_cash / i_cost)
                tmp_cash -= order_result[i] * i_cost
        
        # 重新抓 cond_buy
        cond_buy = (order_result > 0)
        
        # 只有買進時會累加交易成本
        # TODO: 買進 queue, 修正 profit 算法
        self.avr_cost[cond_buy] = (self.avr_cost[cond_buy]*self.position[cond_buy]+self.open[time_index][cond_buy]*order_result[cond_buy])/((self.position+order_result)[cond_buy])
        
        cost = 0
        # append to cost queue
        for i in np.where(cond_buy)[0]:
            i_cost = 0
            for j in range(order_result[i]):
                self.cost_queue[i].append(self.open[time_index][i])
                i_cost += int(self.open[time_index][i] * 1000)
            i_cost += self.fee(i_cost)
            cost += i_cost
        self.cash -= cost
        self.position[cond_buy] += order_result[cond_buy]
        
        unrealized = int(np.sum((self.close[time_index] - self.avr_cost) * self.position * 1000))
        
        # 計算配息
        profit_dividend = np.sum(self.dividend[time_index] * self.position * 1000)
        profit += profit_dividend
        
        self.cnt += 1
        if self.cnt == self.steps:
            self.done = True
        
        return (self.close[self.cnt:self.cnt+self.history_steps], # 歷史收盤價
                self.cash, # 剩餘現金
                unrealized, # 未實現資產市值
                profit, # 損益
                self.avr_cost, # 平均成本
                order, # 原交易單
                order_result, # 實際交易單
                self.position # 持有部位
               )
        
    def fee(self, price):
        if not self.enable_fee:
            return 0
        
        # 手續費 0.1425%
        fee = price * 0.001425
        if fee < 20:
            return 20
        else:
            return int(fee)

if __name__ == '__main__':
    env = Env('./stock_data/')
    
    cash = int(1e+6)
    start_date = '2016/01/25'
    steps = 10
    history_steps = 5
    targets = ['1101', '1301', '2330']
    env.reset(
            cash,
            start_date,
            steps,
            history_steps,
            targets)
    
    actions = [
             [['1101', 1], ['2330', 10]],
             [['1301', 1], ['2330', 2]],
             [['1301', 3], ['2330', -1]],
             [['1101', 5], ['1301', -20]],
             [],
             [],
             [],
             [],
             [],
             [],
            ]
    
    for i in range(steps):
        print('[step %d]' % (i + 1))
        print()
        price, cash, unrealized, profit, avr_cost, order, order_result, position = env.step(actions[i])
        print('target:', targets)
        print('avr_cost:', avr_cost)
        print('order:', order, order_result)
        print('positon:', position)
        
        print('%s\t%s\t%s' % ('cash', 'unrealized', 'profit'))
        print('%d\t%d\t%d' % (cash, unrealized, profit))
        print()