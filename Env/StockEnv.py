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
        self.enable_fee = False
        
    # Constants
    TAX_RATE = 0.003
    FEE_RATE = 0.001425
    FEE_MIN = 20
    
    # 讀取台股交易日
    def load_trading_day(self):
        df = pd.read_excel(self.stock_folder + 'Y9999.xlsx')
        start_date_row = df.loc[df['年月日'] == self.start_date]
        assert len(start_date_row) > 0, '起始日無交易'
        start_date_index = start_date_row.index[0]
        assert start_date_index >= self.history_steps, '交易日資料不足'
        head_date_index = start_date_index - self.history_steps
        end_date_index = start_date_index + self.steps
        self.trading_day = pd.DatetimeIndex(df['年月日'].iloc[head_date_index:end_date_index])
        
    # 讀取個股資料
    def load_target_price(self):
        open_price = pd.DataFrame(index=self.trading_day, columns=self.stock_targets)
        close_price = pd.DataFrame(index=self.trading_day, columns=self.stock_targets)
        
        for i in range(self.stock_targets_count):
            # create table
            df = pd.DataFrame([])
            
            # handle cross-year action
            for year in range(self.trading_day[0].year, self.trading_day[-1].year + 1):
                filename = "{}{}/{}.xlsx".format(self.stock_folder, self.stock_targets[i], year)
                df_tmp = pd.read_excel(filename, index_col = 0)
                df = df.append(df_tmp)
                
            # set date as index
            df = df.reset_index().drop(['index'], axis=1)
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df = df.set_index('Date')
            
            # handle missing prices
            df_slice =  df.loc[self.trading_day, ['Open', 'Close']]
            for index, row in pd.isnull(df_slice).iterrows():
                if row['Open']:
                    df_slice.loc[index, 'Open'] = df.loc[:index, 'Close'].iloc[-1]
                if row['Close']:
                    df_slice.loc[index, 'Close'] = df.loc[:index, 'Close'].iloc[-1]
            
            # update table
            open_price[self.stock_targets[i]] = df_slice['Open']
            close_price[self.stock_targets[i]] = df_slice['Close']
        
        self.open = open_price.values
        self.close = close_price.values
        
    # 讀取除息資料    
    def load_target_dividend(self): 
        df = pd.read_csv(self.stock_folder + 'dividend.csv')
        df['年月日'] = pd.to_datetime(df['年月日'])
        df = df.set_index('年月日')
        
        # create table
        self.dividend = pd.DataFrame(
                data = 0,
                index=self.trading_day, 
                columns=self.stock_targets)
        
        # update table with dividend data from intersection of data cloumns and stock targets
        self.dividend[df.columns & self.stock_targets] = df[df.columns & self.stock_targets]
            
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
        self.stock_targets = stock_targets
        self.stock_targets_count = len(stock_targets)
        self.position = np.zeros(self.stock_targets_count, dtype = int)
        self.avg_cost = np.zeros(self.stock_targets_count)
        self.cost_queue = [deque([]) for _ in range(self.stock_targets_count)]
        
        self.load_trading_day()
        self.load_target_price()
        self.load_target_dividend()
        
        self.stock_targets_idx = {j:i for i, j in enumerate(self.stock_targets)}
        
        return (self.close[:self.history_steps], # 歷史收盤價
                self.cash, # 剩餘現金
                0, # 未實現資產市值
                0, # 損益
                self.avg_cost, # 平均成本
                np.zeros(self.stock_targets_count), # 委託
                np.zeros(self.stock_targets_count), # 成交
                self.position)
        
    def get_fee(self, price):
        if not self.enable_fee:
            return 0
        
        # 手續費 0.1425%
        fee = price * self.FEE_RATE
        if fee < self.FEE_MIN:
            return self.FEE_MIN
        else:
            return int(fee)
        
    # check if sell volume is not larger than position volume
    def __sell_check(self, order):
        # 委賣超過庫存，以庫存量成交
        cond_over_sell = ((self.position + order) < 0)
        order[cond_over_sell] = -1 * self.position[cond_over_sell]
    
    def __sell(self, order, open_price):
        self.__sell_check(order)
        
        cond_sell = (order < 0)
        total_income = 0
        total_cost = 0

        for i in np.where(cond_sell)[0]:
            # 賣出收入
            income = int(open_price[i] * -1 * order[i] * 1000)
            income -= self.get_fee(income) + int(income * self.TAX_RATE)
            total_income += income
            
            # 買進成本
            cost = 0
            for j in range(order[i], 0):
                assert len(self.cost_queue[i]) > 0
                cost += int(self.cost_queue[i].popleft() * 1000)
            cost += self.get_fee(cost)
            total_cost += cost

        profit = total_income - total_cost
        
        # 修正持有部位和平均成本
        self.position[cond_sell] += order[cond_sell]
        self.avg_cost[(self.position) == 0] = 0
        
        return total_income, profit
    
    # check if cash is enough to buy stocks
    def __buy_check(self, order, open_price):
        cond_buy = (order > 0) 
        # 檢查現金是否足夠
        total_cost = np.sum(open_price[cond_buy] * order[cond_buy] * 1000)
        total_cost += self.get_fee(total_cost)
        if self.cash < total_cost:
            # 修改 order[cond_buy]
            tmp_cash = self.cash
            for i in np.where(cond_buy)[0]:
                cost = open_price[i] * 1000
                cost += self.get_fee(cost)
                if (tmp_cash / cost) < order[i]:
                    # 現金不足，計算最大可買張數
                    order[i] = int(tmp_cash / cost)
                tmp_cash -= order[i] * cost
    
    def __buy(self, order, open_price):
        self.__buy_check(order, open_price)
        
        cond_buy = (order > 0)
        
        # 買進時會累加交易成本
        self.avg_cost[cond_buy] = (
                (self.avg_cost * self.position + open_price * order)[cond_buy]
                / (self.position + order)[cond_buy])
        
        total_cost = 0
        # append to cost queue
        for i in np.where(cond_buy)[0]:
            cost = 0
            for j in range(order[i]):
                self.cost_queue[i].append(open_price[i])
                cost += int(open_price[i] * 1000)
            cost += self.get_fee(cost)
            total_cost += cost
            
        self.position[cond_buy] += order[cond_buy]
        return total_cost
    
    def step(self, action):
        date_index = self.history_steps + self.cnt
        
        # 委託單
        order = np.zeros(self.stock_targets_count, dtype = int)
        for code, volume in action:
            order[self.stock_targets_idx[code]] = volume
        # 成交單
        order_deal = order.copy()

        # sell
        income, profit = self.__sell(order_deal, self.open[date_index])
        self.cash += income
        
        # buy
        cost = self.__buy(order_deal, self.open[date_index])
        self.cash -= cost
        
        # 未實現損益
        unrealized = int(np.sum((self.close[date_index] - self.avg_cost) * self.position * 1000))
        
        # 計算配息
        profit_dividend = np.sum(self.dividend.iloc[date_index] * self.position * 1000)
        profit += profit_dividend
        
        self.cnt += 1
        if self.cnt == self.steps:
            self.done = True
        
        return (self.close[self.cnt:date_index], # 歷史收盤價
                self.cash, # 剩餘現金
                unrealized, # 未實現資產市值
                profit, # 損益
                self.avg_cost, # 平均成本
                order, # 委託
                order_deal, # 成交
                self.position)

if __name__ == '__main__':
    env = Env('./stock_data/')
    
    cash = int(1e+5)
    start_date = '2016/01/29'
    steps = 10
    history_steps = 5
    #targets = ['1101', '1301', '2330']
    targets = ['1101', '1102']
    env.reset(
            cash,
            start_date,
            steps,
            history_steps,
            targets)
    
    actions = [
             [['1102', 1]],
             [['1102', 0],['1101', 1]],
             [['1102', 10]],
             [],
             [['1102', -4]],
             [['1101', -1]],
             [],
             [],
             [],
             [],
            ]   
    """
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
    """
    
    for i in range(steps):
        print('[step %d]' % (i + 1))
        print()
        price, cash, unrealized, profit, avg_cost, order, order_deal, position = env.step(actions[i])
        print('target:\t\t', targets)
        print('avg_cost:\t', avg_cost)
        print('order:\t\t', order)
        print('deal:\t\t', order_deal)
        print('positon:\t', position)
        
        print('%s\t%s\t%s' % ('cash', 'profit', 'unrealized'))
        print('%d\t%d\t%d' % (cash, profit, unrealized))
        print()