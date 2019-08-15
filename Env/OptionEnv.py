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
        
    def set_due_date(self, due_date):
        pass
        
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
        self.Y9999 = pd.read_excel(self.option_folder + 'market.xlsx') #大盤
        self.Y9999['年月日'] = pd.to_datetime(self.Y9999['年月日']).apply(lambda x: x.date())
        #保證金AB
        self.margin_AB = pd.read_excel(self.option_folder + 'margin_AB.xlsx')
        self.margin_AB['start'] = pd.to_datetime(self.margin_AB['start']).apply(lambda x: x.date())
        self.margin_AB['end'] = pd.to_datetime(self.margin_AB['end']).apply(lambda x: x.date())
        
        self.data_order = pd.DataFrame(columns=['order_date','order_contract','contract_month'
                                                ,'order_volume'
                                                ,'order_sp','order_CP','open_pm', 'close_pm'
                                                , 'premium'])
       
    
    def reset(self,cash):
        self.pool = 0
#        self.more_money = 0
        self.cash = cash
        self.order_tax = 0.001 #交易稅
        self.order_taxD = 0.00002 #交易稅(結算)
        self.order_fee = 15 #手續費
           
    def step(self, act, date):
        self.open_pm = 0
        position = []
        
        unrealized = 0
        profit = 0
        premium = 0
        loss_margin = 0
        total_profit = 0
        
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
            # TODO: bug
            index = (date.year - 2016) * 12 + date.month - 1   
            if date <= self.dueDate['dueDate'][index]:
                # 結算日前
                contract_month = index + int(order_contract)
            else:
                contract_month = index + int(order_contract)+1

            if (contract_month > 12):
                contract_month = contract_month-12
                
            
            
            #篩選出在資料中是哪一筆data
            target_data = self.data[(self.data['C/P'] == order_CP) & 
                                    (self.data['Strike Price'] == order_sp) & 
                                    (self.data['Date'] == date)]
            #print(target_data)
            #會篩出(近1或近2)兩筆資料、或是當日沒有交易資料
            if (len(target_data) == 0):
                # TODO: 沒有交易資料
                print('No data:', acts)
                continue
            
            #篩出近1和近2的資料
            for a in range (len(target_data)):
                if (target_data['Contract'].iloc[a].month == contract_month): #比對篩出資料的到期月分和data_order的到期月份
                    target_data = target_data.iloc[[a]]
                    self.open_pm = target_data['Open'].iloc[0]
                    self.close_pm = target_data['Close'].iloc[0]
                    break
            target_data = target_data.reset_index()
            
            premium = self.open_pm * 50 * order_volume
            
            
            # 買進
            if order_volume > 0:
                if self.cash < premium:
                    order_volume = -1 * int(self.cash / self.open_pm * 50)
                    premium = self.open_pm * 50 * order_volume
                    
                self.cash -= premium
                position.append([act_type, order_volume,order_sp])
                
            # 賣
            if order_volume < 0:
                '''
                call價外值： MAXIMUM((履約價格-標的價格) ×契約乘數,0)
                put價外值： MAXIMUM((標的價格-履約價格)×契約乘數,0)
                保證金：權利金市值＋MAXIMUM (A值-價外值, B值)
                '''
                self.underlying = self.Y9999[self.Y9999['年月日'] == date]['開盤價(元)'].iloc[0]
                if order_CP == 'C':
                    out_value = np.maximum((order_sp - self.underlying) * 50, 0)
                else:                     
                    out_value = np.maximum((self.underlying - order_sp) * 50, 0)
                
                # 賣方需要的保證金
                margin = np.maximum(self.margin_A_o - out_value, self.margin_B_o) + self.open_pm * 50
                
                # TODO: 計算保證金
                if self.pool < margin * order_volume: # pool中的錢不夠,要求補錢
                    dif = margin * order_volume - self.pool
                    if dif <= self.cash: # 環境夠付
                        self.pool += dif
                        self.cash -= dif
                        position.append([act_type, order_volume, order_sp])
                    else:
                        # 只賣出環境夠付的部位
                        order_volume = -1 * int(self.cash / margin)
                        # 跟環境要錢
                        self.cash = self.cash - order_volume * margin
                        
                        position.append([act_type, order_volume, order_sp])
                else:
                    position.append([act_type, order_volume,order_sp])
                    
            self.data_order = self.data_order.append({
                    'order_date':date,
                    'order_contract':order_contract,
                    'contract_month':contract_month,
                    'order_volume':order_volume,
                    'order_sp':order_sp,
                    'order_CP':order_CP,
                    'open_pm':self.open_pm,
                    'close_pm':self.close_pm,
                    'premium':premium,
                    }, ignore_index=True)

        if date in self.SettDate:
            #判斷是否為結算日    
            sett_p = self.dueDate[self.dueDate['dueDate']==date]['settlement'].iloc[0]

            for j in range (len(self.data_order)):
                #取order中strike price
                price_pre = int(self.data_order['order_sp'][j])
                #到期日前，算profit
                if (self.data_order['order_date'][j]<=date):
                    if (self.data_order['order_volume'][j] > 0) & (self.data_order['order_CP'][j] == 'C'):#buy call
                        profit = (np.maximum((sett_p - price_pre), 0) * 50 * 
                                  int(self.data_order['order_volume'][j]) - self.data_order['premium'][j])* (1 - self.order_taxD)
                    elif (self.data_order['order_volume'][j]>0) & (self.data_order['order_CP'][j] == 'P'):
                        profit = (np.maximum((price_pre-sett_p),0)*50*int(self.data_order['order_volume'][j])-self.data_order['premium'][j])*(1-self.order_taxD)
                    elif (self.data_order['order_volume'][j]<0) & (self.data_order['order_CP'][j] == 'C'):
                        profit = -np.maximum((sett_p-price_pre),0)*50*int(self.data_order['order_volume'][j]*(-1))+self.data_order['premium'][j]
                        loss_margin = (sett_p-price_pre)*50*int(self.data_order['order_volume'][j])*(-1)
                    else:
                        profit = np.maximum((price_pre-sett_p),0)*50*int(self.data_order['order_volume'][j]*(-1))+self.data_order['premium'][j]
                        loss_margin = (price_pre-sett_p)*50*int(self.data_order['order_volume'][j])*(-1)

                    self.data_order = self.data_order.drop([j])
                    total_profit += profit
                     
                self.pool -= loss_margin
                
        else: #buy方未實現損益
            for j in range (len(self.data_order)):
                if(self.data_order['order_volume'][j] > 0):
                    unrealized += (self.data_order['close_pm'][j]-self.data_order['open_pm'][j])*50*int(self.data_order['order_volume'][j])*(1-self.order_tax)
                if(self.data_order['order_volume'][j] < 0):
                    unrealized += (self.data_order['premium'][j])

        self.cash += total_profit
        return self.cash, total_profit, position, unrealized                     
    
        
#%%
if __name__ == '__main__':
    option_folder = './option_data/'
    
    start_date = '2016/01/19'
    period = 4
    action = list([
            [['TXO01_C',1,7900],['TXO02_C',-2,7900],['TXO01_P',-1,7700]],
            [['TXO02_P',2,8700],['TXO01_C',2,7900],['TXO01_P',1,8500]],
            [['TXO01_C',1,9200],['TXO02_C',-1,9200]],
            [['TXO01_P',-1,7700]]
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
        print(date)
        cash, profit, position, unrealized = env.step(act, date)
        
        print(cash, profit, position, unrealized)
    #%%
    dueDate = env.dueDate
    Data = env.data
#    env.reset(cash)
    data_order = env.data_order        

    margin_AB = env.margin_AB

#%%
#追繳保證金但env不夠時應該全部平倉還是部份
    
    
    