# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:50:19 2019

@author: feifanhe
"""
#%%
import numpy as np
from collections import deque
import StockEnv

#%%
class Env(StockEnv.Env):
    def __init__(
            self,
            stock_folder):
        super().__init__(stock_folder)

    def reset(
            self,
            repeat,
            cash, #初始化資金
            start_date, #交易起始日
            steps, #交易天數
            history_steps, #窺視歷史價格天數
            stock_targets, #要進行資產配置的股票
            ):
        # set attributes
        super().reset(
            cash,
            start_date,
            steps,
            history_steps,
            stock_targets)
        self.repeat = repeat
        self.cash = np.array([cash] * self.repeat, dtype = int)
        
        # reset position
        self.position_shape = (self.repeat, self.stock_targets_count)
        self.position = np.zeros(self.position_shape)
        self.cost_queue = [[deque([]) for _ in range(self.stock_targets_count)] for _ in range(self.repeat)]
        
        self.open = np.tile(self.open, (self.repeat, 1, 1))
        self.close = np.tile(self.close, (self.repeat, 1, 1))
        self.dividend = np.tile(self.dividend, (self.repeat, 1, 1))
        self.ex_right = np.tile(self.ex_right, (self.repeat, 1, 1))
        
        self.done = False
        
        return (self.close[:, :self.history_steps], # 歷史收盤價
                self.cash)

    def __sell(self, order, open_price):
        self.__sell_check(order)
        
        cond_sell = order < 0
        order_deal = order.copy()
        order_deal[np.logical_not(cond_sell)] = 0

        # determine total income and profit
        total_income = np.sum(open_price * -1 * order_deal * 1000, axis = 1).astype(int)
        total_income -= self.get_fee(total_income) + (total_income * self.TAX_RATE).astype(int)
        
        total_cost = np.zeros(self.repeat, dtype = int)
        for i in range(self.repeat):
            for target in np.where(cond_sell[i])[0]:
                # 買進成本
                cost = 0
                for k in range(order_deal[i, target], 0):
                    assert len(self.cost_queue[i][target]) > 0
                    cost += int(self.cost_queue[i][target].popleft() * 1000)
                cost += self.get_fee(cost)
                total_cost[i] += cost

        profit = total_income - total_cost
        
        # 修正持有部位和平均成本
        self.position += order_deal
        
        return total_income, profit
    
    # check if cash is enough to buy stocks
    def __buy_check(self, order_deal, cond, open_price):
        total_cost = np.sum(open_price * order_deal * 1000, axis = 1).astype(int)
        total_cost += self.get_fee(total_cost)
        
        cond_not_enough = self.cash < total_cost
        
        for i in np.where(cond_not_enough)[0]:
            tmp_cash = self.cash[i]
            for j in np.where(cond[i])[0]:
                cost = open_price[i, j] * 1000
                cost += self.get_fee(cost)
                if (tmp_cash / cost) < order_deal[i, j]:
                    order_deal[i, j] = int(tmp_cash / cost)
                tmp_cash -= order_deal[i, j] * cost
            
    def __buy(self, order, open_price):
        cond_buy = order > 0
        order_deal = order.copy()
        order_deal[np.logical_not(cond_buy)] = 0
        total_cost = np.zeros(self.repeat, dtype = int)
        
        self.__buy_check(order_deal, cond_buy, open_price)
        order[cond_buy] = order_deal[cond_buy]
        
        # append to cost queue
        for i in range(self.repeat):
            for j in np.where(cond_buy[i])[0]:
                self.cost_queue[i][j].extend([open_price[i, j]] * order_deal[i, j])
            
        total_cost = np.sum(open_price * order_deal * 1000, axis = 1).astype(int)
        total_cost += self.get_fee(total_cost)
                
        self.position += order_deal
        return total_cost
    
    def __parse_order(self, actions):
        # 委託單
        order = np.zeros(self.position_shape, dtype = int)
        for i, action in enumerate(actions):
            for code, volume in action:
                order[i, self.stock_targets_idx[code]] = volume
        return order
    
    def step(self, actions):
        date_index = self.history_steps + self.cnt
        
        # 委託單
        order = self.__parse_order(actions)
        order_deal = order.copy()

        # sell
        income, profit = self.__sell(order_deal, self.open[:, date_index])
        self.cash += income
        
        # buy
        cost = self.__buy(order_deal, self.open[:, date_index])
        self.cash -= cost
        
        # average cost
        # TODO: seperate as a function
        avg_cost = np.zeros(self.position_shape, dtype = float)
        
        for i in range(self.repeat):
            for j in range(self.stock_targets_count):
                if self.position[i, j] == 0:
                    avg_cost[i] = 0
                    continue
                avg_cost[i, j] = sum(self.cost_queue[i][j]) / self.position[i, j]
            
        # 未實現損益
        unrealized = np.sum((self.close[:, date_index] - avg_cost) * self.position * 1000, axis = 1)
        
        # 計算配息
        profit_dividend = np.sum(self.dividend[:, date_index] * self.position * 1000, axis = 1).astype(int)
        profit += profit_dividend
        
        # 計算配股
        self.position += self.position * self.ex_right[:, date_index] / 10
        
        self.cnt += 1
        if self.cnt == self.steps:
            self.done = True
        
        return (self.close[:, self.cnt:date_index + 1], # 歷史收盤價
                self.cash, # 剩餘現金
                unrealized, # 未實現資產市值
                profit, # 損益
                avg_cost, # 平均成本
                order, # 委託
                order_deal, # 成交
                self.position)

if __name__ == '__main__':
    env = Env('./stock_data/')
    
    repeat = 5
    cash = int(1e+5)
    start_date = '2016/01/29'
    steps = 3
    history_steps = 5
    targets = ['1101', '1102']
    
    price, cash = env.reset(
            repeat,
            cash,
            start_date,
            steps,
            history_steps,
            targets)
    
    actions = [
             [[['1102', 1]]],
             [[['1102', 0],['1101', 1]]],
             [[['1102', 10]]],
             [[]],
             [[['1102', -4]]],
             [[['1101', -1]]],
             [[]],
             [[]],
             [[]],
             [[]],
            ]
    
    for i in range(steps):
        print('[step %d]' % (i + 1))
        print()
        price, cash, unrealized, profit, avg_cost, order, order_deal, position = env.step(actions[i])
        print('target:\t', targets)
        print('avg_cost:\n', avg_cost)
        print('order:\n', order)
        print('deal:\n', order_deal)
        print('positon:\n', position)
        
        print(f'Cash:\n{cash}\nProfit:\n{profit}\nUnrealize:\n{unrealized}')
        print()