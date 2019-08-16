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
        
        self.stock_targets_idx = {j:i for i,j in enumerate(self.stock_targets)}
        
        return (self.close[:self.history_steps], # 歷史收盤價
                self.cash, # 剩餘現金
                0, # 未實現資產市值
                0, # 損益
                self.avg_cost, # 平均成本
                np.zeros(self.stock_targets_count), # 委託
                np.zeros(self.stock_targets_count), # 成交
                self.position # 庫存
               )
        
    def step(self, action):
        date_index = self.history_steps + self.cnt
        # 委託單
        order = np.zeros(self.stock_targets_count, dtype = int)
        for code, volume in action:
            order[self.stock_targets_idx[code]] = volume
        
        # 成交單
        order_deal = order.copy()

        # 委賣超過庫存，以庫存量成交
        cond_over_sell = ((self.position + order) < 0)
        order_deal[cond_over_sell] = -1 * self.position[cond_over_sell]

        # 賣出
        cond_sell = (order_deal < 0)  
        
        # 賣出收入
        income = 0
        for i in np.where(cond_sell)[0]:
            i_income = int(self.open[date_index, i] * -1 * order_deal[i] * 1000)
            i_income -= self.get_fee(i_income)
            income += i_income
        income -= int(income * self.TAX_RATE)

        # 買進成本
        cost = 0
        for i in np.where(cond_sell)[0]:
            i_cost = 0
            for j in range(order_deal[i], 0):
                assert len(self.cost_queue[i]) > 0
                i_cost += int(self.cost_queue[i].popleft() * 1000)
            i_cost += self.get_fee(i_cost)
            cost += i_cost
            
        profit = income - cost
        self.cash += int(income)
        
        # 修正持有部位和平均成本
        self.position[cond_sell] += order_deal[cond_sell]
        self.avg_cost[((self.position)==0)] = 0
        
        # 買入
        cond_buy = (order_deal > 0) 

        # 檢查現金是否足夠
        cost = np.sum(self.open[date_index, cond_buy] * order_deal[cond_buy] * 1000)
        cost += self.get_fee(cost)
        if self.cash < cost:
            # 修改 order_deal[cond_buy]
            tmp_cash = self.cash
            for i in np.where(cond_buy)[0]:
                i_cost = self.open[date_index][i] * 1000
                i_cost += self.get_fee(i_cost)
                if (tmp_cash / i_cost) < order_deal[i]:
                    # 現金不足，計算最大可買張數
                    order_deal[i] = int(tmp_cash / i_cost)
                tmp_cash -= order_deal[i] * i_cost
        
        # 重新抓 cond_buy
        cond_buy = (order_deal > 0)
        
        # 買進時會累加交易成本
        self.avg_cost[cond_buy] = ((self.avg_cost * self.position +
                                   self.open[date_index] * order_deal)[cond_buy]
                                  / (self.position + order_deal)[cond_buy])
        
        cost = 0
        # append to cost queue
        for i in np.where(cond_buy)[0]:
            i_cost = 0
            for j in range(order_deal[i]):
                self.cost_queue[i].append(self.open[date_index][i])
                i_cost += int(self.open[date_index][i] * 1000)
            i_cost += self.get_fee(i_cost)
            cost += i_cost
        self.cash -= cost
        
        self.position[cond_buy] += order_deal[cond_buy]
        
        unrealized = int(np.sum((self.close[date_index] - self.avg_cost) * self.position * 1000))
        
        # 計算配息
        profit_dividend = np.sum(self.dividend.iloc[date_index] * self.position * 1000)
        profit += profit_dividend
        
        self.cnt += 1
        if self.cnt == self.steps:
            self.done = True
        
        return (self.close[self.cnt:self.cnt+self.history_steps], # 歷史收盤價
                self.cash, # 剩餘現金
                unrealized, # 未實現資產市值
                profit, # 損益
                self.avg_cost, # 平均成本
                order, # 委託
                order_deal, # 成交
                self.position # 庫存
               )
        
    def get_fee(self, price):
        if not self.enable_fee:
            return 0
        
        # 手續費 0.1425%
        fee = price * self.FEE_RATE
        if fee < self.FEE_MIN:
            return self.FEE_MIN
        else:
            return int(fee)

if __name__ == '__main__':
    env = Env('./stock_data/')
    
    cash = int(1e+6)
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
             [['1102', 1]],
             [],
             [['1102', -2]],
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