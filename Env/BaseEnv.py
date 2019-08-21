#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:48:09 2019

@author: feifanhe
"""

#%%
import pandas as pd
import numpy as np
import StockEnv as stock
import FuturesEnv as futures
import OptionEnv as option

#%%
class BaseEnv():
    def __init__(
            self,
            stock_folder,
            futures_folder,
            option_folder,
            ):
        self.stock_folder = stock_folder
        self.done = True
        
        self.env_stock = stock.Env(stock_folder)
        
        self.env_futures = futures.Env(futures_folder)
        
        self.env_option = option.Env(option_folder)
        self.env_option.load()
        
    def reset(
            self,
            cash,
            start_date,
            steps,
            history_steps,
            targets,
            ):
        self.counter = 0
        self.cash = cash
        self.start_date = start_date
        self.steps = steps
        self.history_steps = history_steps
        
        # deal with targets
        self.targets = np.array(targets)
        self.stock_targets = self.targets[self.targets[:, 0] == 's', 1]
        self.futures_targets = self.targets[self.targets[:, 0] == 'f', 1]
        self.option_targets = self.targets[self.targets[:, 0] == 'o', 1]
        
        # reset environments
        self.env_stock.reset(
                self.cash,
                self.start_date,
                self.steps,
                self.history_steps,
                self.stock_targets)
        self.env_futures.reset(
                self.cash,
                self.start_date,
                self.steps,
                self.history_steps)
        self.env_option.reset(self.cash)
        
        self.done = False
    
    def step(
            self,
            actions,
            ):
        
        stock_actions = []
        futures_actions = []
        option_actions = []
        for action in actions:
            if action[0] == 's':
                stock_actions.append(action[1:])
            if action[0] == 'f':
                futures_actions.append(action[1:])
            if action[0] == 'o':
                option_actions.append(action[1:])
        
        self.env_stock.cash = self.cash
        
        # stock
        print('ENV_STOCK STEP:', stock_actions)
        price, cash, unrealized, profit, avr_cost, order, order_result, position = self.env_stock.step(stock_actions)
        print('target:', self.stock_targets)
        print('avr_cost:', avr_cost)
        print('order:', order, '->', order_result)
        print('positon:', position)
        
        print('%s\t%s\t%s' % ('cash', 'profit', 'unrealized'))
        print('%d\t%d\t%d\n' % (cash, profit, unrealized))
        
        self.cash = cash
        self.env_futures.cash = self.cash
        
        # future
        print('ENV_TX STEP:', futures_actions)
        cash, unrealized, profit, position, avg_cost, order, deal, margin_call = self.env_futures.step(futures_actions)
        print(f'Pos:\t{position}')
        print(f'Cost:\t{avg_cost}')
        print('cash \t profit \t margin \t unrealized')
        print(f'{cash}\t{profit}\t{margin_call}\t{unrealized}\n')
        
        self.cash = cash
        self.env_option.cash = self.cash
        
        # option
        print('ENV_TXO STEP:', option_actions)
        trading_day = self.env_stock.trading_day
        date = pd.to_datetime(trading_day[self.history_steps + self.counter]).date()
        cash_o, profit, position, unrealized = self.env_option.step(option_actions, date)
        print('%s\t%s\t%s' % ('cash', 'profit', 'unrealized'))
        print('%d\t%d\t%d' % (cash_o, profit, unrealized))
        print('\n================\n')
        
        self.cash = cash_o
        
        self.counter += 1
        
#%%

if __name__ == '__main__':
    stock_folder = './stock_data/'
    futures_folder = './futures_data/'
    option_folder = './option_data/'
    base_env = BaseEnv(stock_folder, futures_folder, option_folder)
    
    cash = int(3e+6)
    start_date = '2016/01/19'
    steps = 4
    history_steps = 5
    targets = [['s', '1101'],
               ['s', '1301'],
               ['s', '2330'],
               ['f', 'TX01'],
               ['f', 'TX02'],
               ['f', 'MTX01'], # TX01. TX02, MTX01, MTX02
              ]
    
    base_env.reset(
            cash,
            start_date,
            steps,
            history_steps,
            targets)
    
    actions = [
             # action 0
             [['s', '1101', 1], ['s', '2330', 10],
              ['f', 'TX01',1], ['f', 'TX02',1], ['f', 'MTX01',1],
              ['o', 'TXO01_C', 1, 7900], ['o', 'TXO02_C', -2, 7900], ['o', 'TXO01_P', -1, 7700]],
             # action 1
             [['s', '1301', 1], ['s', '2330', 2],
              ['f', 'TX01',-1], ['f', 'TX02',-3],
              ['o', 'TXO02_P', 2, 8700], ['o', 'TXO01_C', 2, 7900], ['o', 'TXO01_P', 1, 8500]],
             # action 2
             [['s', '1301', 3], ['s', '2330', -1],
              ['f', 'TX01',1], ['f', 'TX02',2],
              ['o', 'TXO01_C', 1, 9200], ['o', 'TXO02_C', -1, 9200]],
             # action 3
             [['s', '1101', 5], ['s', '1301', -20],
              ['o', 'TXO01_P', -1, 7700]]
            ]

    
    for i in range(steps):
        if base_env.done:
                break
        
        print('[step %d]\n' % (i + 1), actions[i], '\n')
        base_env.step(actions[i])
