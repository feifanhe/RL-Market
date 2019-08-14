# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:22:17 2019

@author: chocy
"""
import pandas as pd
import numpy as np
#%%
class Env:
    def __init__(self, option_folder):
        # Env parameter initial
        self.option_folder = option_folder
        
    def load(self):
        #選擇權dataset
        self.data = pd.read_csv(self.option_folder + 'txo_2016_2018_new.csv')
        self.data = self.data.drop(columns=['Unnamed: 0'])
        #轉成datetime
        self.data['Date'] = pd.to_datetime(self.data['Date']).apply(lambda x: x.date())
        self.data['Contract'] = pd.to_datetime(self.data['Contract']).apply(lambda x: x.date())
        #到期日dataset
        self.dueDate = pd.read_csv(self.option_folder + 'DueDate_txo.csv')
        self.dueDate['dueDate'] = pd.to_datetime(self.dueDate['dueDate']).apply(lambda x: x.date())
        self.SettDate = self.dueDate['dueDate'].tolist()
#        self.dueDate['dueDate'] = pd.to_datetime(self.dueDate['dueDate']).apply(lambda x: x.date())
#        self.dueDate=self.dueDate_price['dueDate'].tolist()
        self.data_market = pd.read_excel(self.option_folder + 'market.xlsx') #大盤
        self.data_market['年月日'] = pd.to_datetime(self.data_market['年月日']).apply(lambda x: x.date())
        #保證金AB
        self.margin_AB = pd.read_excel(self.option_folder + 'margin_AB.xlsx')
        self.margin_AB['start'] = pd.to_datetime(self.margin_AB['start']).apply(lambda x: x.date())
        self.margin_AB['end'] = pd.to_datetime(self.margin_AB['end']).apply(lambda x: x.date())
        
        self.data['Contract_type'] = 0
        self.data_order = pd.DataFrame(columns=['order_date','order_contract','contract_m'
                                                ,'order_volume'
                                                ,'order_sp','order_CP','open_p', 'close_p'
                                                , 'premium'])
       
    
    def reset(self,cash):
        self.data_order = pd.DataFrame(columns=['order_date','order_contract','order_volume'
                                                'order_sp','order_CP'])
        self.need_m = 0
        self.need_o = 0
        self.op_pool = 0
#        self.more_money = 0
        self.cash = cash
           
    def step(self, act, date):    

        order_date = date
        index = date.month-1
        self.data_s = pd.DataFrame(columns=['Symbol','Contract', 'C/P', 'Strike Price', 'Date'
                                            , 'Open', 'High', 'Low', 'Close', 'Volume'])    
        self.market = pd.DataFrame()
        self.open_p = 0
        position = []
        
        order_tax = 0.001 #交易稅(非交易日動作)
        order_taxD = 0.00002 #交易稅(交易日動作)
        order_fee = 15 #手續費
        unrealized = 0
        profit = 0
        premium = 0
        loss_margin = 0
        
        if date in self.data.Date.values:
            for index, row in self.margin_AB.iterrows():       
                start = pd.to_datetime(row['start']).date()
                end = pd.to_datetime(row['end']).date()
                if start <= date <= end:
                    self.margin_A_o = row['o_margin_a']
                    self.margin_A_m = row['m_margin_a']
                    self.margin_B_o = row['o_margin_b']
                    self.margin_B_m = row['m_margin_b']
                    break
                                    
        for acts in act:
            self.data_s = self.data.copy()
            act_type = acts[0]
            order_volume = acts[1]
            order_sp = acts[2]
            total_profit =0
            
            if act_type == 'TXO01_C':
                order_contract = '1'
                order_CP = 'C'
            elif act_type == 'TXO02_C':
                order_contract = '2'
                order_CP = 'C'
            elif act_type == 'TXO01_P':
                order_contract = '1'
                order_CP = 'P'
            else:
                order_contract = '2'
                order_CP = 'P'
            
            #找出到期日月份
            if(date <= self.dueDate['dueDate'][index]):
                contract_m = index + int(order_contract)
            else:
                contract_m = index + int(order_contract)+1
            #超過12月就減12
            if(contract_m > 12):
                contract_m = contract_m-12
                      
            #篩選出在資料中是哪一筆data
            self.data_s = self.data[(self.data_s['C/P'] == order_CP) & (self.data['Strike Price'] == int(order_sp)) & (self.data['Date'] == str(order_date))]
            #會篩出(近1或近2)兩筆資料、或是當日沒有交易資料

            if (len(self.data_s)==0):
            #       #尚未從原有資料找到，從大盤中抓開盤價
                self.market = self.data_market[self.data_market['年月日'] == order_date]
                self.open_p = (self.market['開盤價(元)'].iloc[0])/50
                self.close_p = (self.market['收盤價(元)'].iloc[0])/50
            else: #篩出近1和近2的資料
                for a in range (len(data_s)):
                    if(self.data_s['Contract'].iloc[a].month == contract_m): #比對篩出資料的到期月分和data_order的到期月份
                        self.data_s = self.data_s.iloc[[a]]
                        self.open_p = self.data['Open'].iloc[0]
                        self.close_p = self.data['Close'].iloc[0]
                        break
            self.data_s = self.data_s.drop(self.data_s.index,inplace=True)
            
           
            premium =  (self.open_p*50*order_volume+order_fee)*(1-order_tax)
            if(order_volume >0):
                premium = -(self.open_p*50*order_volume+order_fee)*(1-order_tax)
            else:
                premium = (-self.open_p*50*order_volume+order_fee)*(1-order_tax)
                
            self.data_order = self.data_order.append({'order_date':order_date, 'order_contract':order_contract, 'contract_m':contract_m
                                                      , 'order_volume':order_volume
                                                      , 'order_sp':order_sp, 'order_CP':order_CP, 'open_p':self.open_p
                                                      , 'close_p':self.close_p, 'premium':premium}, ignore_index=True)
            #判斷是賣
            if order_volume <0:
                '''
                call價外值： MAXIMUM((履約價格-標的價格) ×契約乘數,0)
                put價外值： MAXIMUM((標的價格-履約價格)×契約乘數,0)
                保證金：權利金市值＋MAXIMUM (A值-價外值, B值)
                '''
                self.underlying = self.data_market[self.data_market['年月日'] == order_date]
                if order_CP == 'C':
                    out_value = np.maximum((order_sp*50-self.underlying['開盤價(元)'].iloc[0]),0)
                else:                     
                    out_value = np.maximum((self.underlying['開盤價(元)'].iloc[0]-order_sp*50),0)
                
                need_premium = (premium + out_value)*(order_volume)*(-1) #賣方需要的保證金
                
                self.need_m = self.need_m + self.need_m*order_volume*(-1) #維持保證金
                self.need_o = self.need_o + self.need_o*order_volume*(-1) #原始保證金 
                
                self.min_money = need_premium + self.need_o
               
                if self.op_pool < self.min_money: #pool中的錢不夠,要求補錢
                    dif = self.min_money - self.op_pool
                    if dif <= self.cash: #環境夠付
                        self.op_pool = self.op_pool+dif
                        self.cash = self.cash-dif
                        position.append([act_type, order_volume, order_sp])
                    else:
                        sell_vol = int(self.cash/need_premium)
                        position.append([act_type, sell_vol, order_sp])
                        #跟環境要錢
                        self.cash = self.cash-sell_vol*need_premium
                        order_volume = sell_vol*(-1)
                else:
                    position.append([act_type, order_volume,order_sp])
                
    
        if date in self.SettDate:
            #判斷是否為結算日           
            sett_p = self.dueDate[self.dueDate['dueDate']==date]

            for j in range (len(self.data_order)):
                #取order中strike price
                price_pre = int(self.data_order['order_sp'][j])
                #到期日前，算profit
                if (self.data_order['order_date'][j]<=date):
                    if (self.data_order['order_volume'][j]>0) & (self.data_order['order_CP'][j] == 'C'):#buy call
                        profit = (np.maximum((sett_p['settlement'][0]-price_pre),0)*50*int(self.data_order['order_volume'][j])-self.data_order['premium'][j])*(1-order_taxD)
                    elif (self.data_order['order_volume'][j]>0) & (self.data_order['order_CP'][j] == 'P'):
                        profit = (np.maximum((price_pre-sett_p['settlement'][0]),0)*50*int(self.data_order['order_volume'][j])-self.data_order['premium'][j])*(1-order_taxD)
                    elif (self.data_order['order_volume'][j]<0) & (self.data_order['order_CP'][j] == 'C'):
                        profit = -np.maximum((sett_p['settlement'][0]-price_pre),0)*50*int(self.data_order['order_volume'][j]*(-1))+self.data_order['premium'][j]
                        loss_margin = (sett_p['settlement'][0]-price_pre)*50*int(self.data_order['order_volume'][j])*(-1)
                    else:
                        profit = np.maximum((price_pre-sett_p['settlement'][0]),0)*50*int(self.data_order['order_volume'][j]*(-1))+self.data_order['premium'][j]
                        loss_margin = (price_pre-sett_p['settlement'][0])*50*int(self.data_order['order_volume'][j])*(-1)

                    self.data_order['order_volume'][j]=0
                    total_profit = total_profit + profit
 #                   print (str(self.data_order['order_date'][j])+" : "+str(profit))
                '''                
                else:  
                    if(self.data_order['order_volume'][j] > 0):
                        unrealized = (self.data_order['close_p'][j]-self.data_order['open_p'][j])*50*int(self.data_order['order_volume'][j])*(1-order_tax)
                    else:
                        unrealized = str(self.data_order['premium'][j])+" (may be more loss)"
                    print (str(self.data_order['order_date'][j])+" : "+str(unrealized))
                '''                        
                self.min_money -= loss_margin
                
        else:#buy方未實現損益
            '''
            for j in range (len(self.data_order)):
                if(self.data_order['order_date'][j]>date):
                    if(self.data_order['order_volume'][j] > 0):
                        unrealized = (self.data_order['close_p'][j]-self.data_order['open_p'][j])*50*int(self.data_order['order_volume'][j])*(1-order_tax)
                    else:
                        unrealized = str(self.data_order['premium'][j])+" (may be more loss)"
                    print (str(self.data_order['order_date'][j])+" : "+str(unrealized))
            '''  
            for j in range (len(self.data_order)):
                
                if(self.data_order['order_volume'][j] > 0):
                    unrealized = (self.data_order['close_p'][j]-self.data_order['open_p'][j])*50*int(self.data_order['order_volume'][j])*(1-order_tax)
 #                   print (str(self.data_order['order_date'][j])+" : "+str(unrealized))                      
                if(self.data_order['order_volume'][j] < 0):
                    unrealized = (self.data_order['premium'][j])
#                    print (str(self.data_order['order_date'][j])+" : "+str(unrealized)) 

        return self.cash, profit, premium, position, unrealized                     
    
        
#%%
if __name__ == '__main__':
    option_folder = './option_data/'
    
    start_date = '2016/1/19'
    period = 4
    action = list([
            [['TXO01_C',1,8500],['TXO02_C',-2,8200],['TXO01_P',-1,9300]],
            [['TXO02_P',2,8700],['TXO01_C',2,7900],['TXO01_P',1,8500]],
            [['TXO01_C',1,9200],['TXO02_C',-1,9200]],
            [['TXO01_P',-1,8900]]
            ])   
    
    cash = 1e+6
    env = Env(option_folder)  
    env.load()
    env.reset(cash)
    
    for i in range(period):
        act = action[i]
        s_date = pd.to_datetime(start_date)
        # TODO: [to 非凡] 用 DataOffset 算日期會遇到+1，結果不在資料裡，改成你現在計算日期的方式即可
        date = s_date + pd.DateOffset(days=i)
        date = date.date()
        env.step(act, date)
        
        cash, profit, premium, position, unrealized = env.step(act, date)
        
        print(cash, profit, premium, position, unrealized)
    #%%
    dueDate = env.dueDate
    Data = env.data
#    env.reset(cash)
    data_order = env.data_order        
    data_s = env.data_s
    data_market = env.data_market
    margin_AB = env.margin_AB

#%%
#追繳保證金但env不夠時應該全部平倉還是部份
    
    
    