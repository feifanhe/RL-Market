#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:23:30 2019

@author: feifanhe
"""

import Env.StockEnv as stock

if __name__ == '__main__':
    env = stock.Env('./Env/stock_data/')
    
    cash = int(1e+5)
    start_date = '2016/01/29'
    steps = 10
    history_steps = 5
    targets = ['1101', '1102']
    
    price, cash = env.reset(
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