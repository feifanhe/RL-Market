# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:22:17 2019

@author: yawen
"""
import pandas as pd
#%%

class Env:
    def __init__(self, futures_folder):
        # Env parameter initial
        self.futures_folder = futures_folder
        
    def load(self):
        #大台dataset
        data = pd.read_csv(self.futures_folder + 'tx_2016_2018_new.csv')
        data = data.drop(columns=['Unnamed: 0'])
        data['Date'] = pd.to_datetime(data['Date']).apply(lambda x: x.date())
        #小台dataset
        mdata = pd.read_csv(self.futures_folder + 'mtx_2016_2018_new.csv')
        mdata = mdata.drop(columns=['Unnamed: 0'])
        mdata['Date'] = pd.to_datetime(mdata['Date']).apply(lambda x: x.date())
        #到期日dataset
        self.dueDate_price = pd.read_csv(self.futures_folder + 'DueDate.csv')
        self.dueDate_price['dueDate'] = pd.to_datetime(self.dueDate_price['dueDate']).apply(lambda x: x.date())
        self.dueDate=self.dueDate_price['dueDate'].tolist()
        
        self.maintain_margin = pd.read_csv(self.futures_folder + 'maintain_margin.csv') #保證金
        #%%
        #把近1近2標出來
        data['contract_rank']=data.groupby('Date')['Contract'].rank()
        mdata['contract_rank']=mdata.groupby('Date')['Contract'].rank()
        #大台小台dataset合併
        self.data = pd.concat([data, mdata])

    def reset(self, cash):
        self.own = pd.DataFrame(columns=['type','due_mon','buy_point','volumn','per_price'])
        self.m_acct = 0
        self.o_acct = 0
        self.pool = 0
        self.more_money = 0
        self.cash = cash

#%%
    def buy(self,date,buy_num,act_type,o_margin,m_margin,tx_type,due_mon,per_mon):   
        cost = 0 
        position = []
        abs_buy_num = abs(buy_num)
        is_pos = buy_num / abs_buy_num #正數還是負數 
        #向env要原始保證金的錢，先看環境的錢夠不夠給
        b_money = abs_buy_num*o_margin
        min_money = b_money + self.o_acct
        
        if self.pool< min_money:  # pool裡的錢不夠付保證金  
            dif = min_money - self.pool #dif是當pool中還差多少就到原始保證金，dif一定會>0 若<0就不會進到這
            if dif <= self.cash : #若環境夠付
                self.pool = self.pool + dif
                self.cash = self.cash - dif
    
            #若環境錢不夠給，看最多可以給多少                
            else:
                buy_vol = int(self.cash / o_margin)
                #跟env要錢
                self.cash = self.cash - buy_vol*o_margin
                self.pool = self.pool + buy_vol*o_margin
                #把之後要用到buy_num改成新的
                buy_num = buy_vol*is_pos
                abs_buy_num = abs(buy_num)
        #position append                    
        position=[act_type, buy_num]

        #算買的cost
        c_point = self.data[(self.data['Date']==date) & (self.data['contract_rank']==due_mon) & (self.data['Symbol']==tx_type)]['Open'].iloc[0]
        cost = o_margin*buy_num*is_pos
        #存到own中
        s1 = pd.Series({'type':tx_type, 'due_mon':due_mon,'buy_point':c_point, 'volumn':buy_num, 'per_price':per_mon})
        self.own = self.own.append(s1, ignore_index=True)
        #成交稅:成交價的0.00002 = 點數直接除250 ，每口都要算
        #cost = cost + (c_point/250)*buy_num*is_pos
        #持有的口數的最低原始與維持保證金為何
        self.m_acct = self.m_acct + m_margin*abs_buy_num #帳戶中的維持保證金需為
        self.o_acct = self.o_acct + o_margin*abs_buy_num #帳戶中的原始保證金需為
        #把數量==0的或從own中刪掉
        self.own = self.own[self.own['volumn']!=0]
        self.own = self.own.reset_index(drop=True)
        
        return cost, position
                    
    #%%    
    def sell(self,date,buy_num,act_type,o_margin,m_margin,tx_type,due_mon,per_mon):
        profit = 0
        cost = 0
        position = []
        abs_buy_num = abs(buy_num)
        is_pos = buy_num / abs_buy_num #正數還是負數     
        
        p_point = self.data[(self.data['Date']==date) & (self.data['contract_rank']==due_mon) & (self.data['Symbol']==tx_type)]['Open'].iloc[0]                  
        t_b_num = abs_buy_num              
        for w in range(len(self.own)):
            if (self.own['type'][w] == tx_type) & (self.own['due_mon'][w] == due_mon) :
                if t_b_num <= abs(self.own['volumn'][w]):
                    self.own['volumn'][w] = self.own['volumn'][w] + t_b_num*is_pos
                    this_profit = (p_point - self.own['buy_point'][w])*t_b_num*per_mon*is_pos
                    profit = profit + this_profit
                    #pool增加
                    self.pool = self.pool + this_profit
                    break
                else:
                    t_b_num = t_b_num - abs(self.own['volumn'][w])
                    this_profit = (p_point - self.own['buy_point'][w])*abs(self.own['volumn'][w])*per_mon*is_pos
                    profit = profit + this_profit
                    #pool增加
                    self.pool = self.pool + this_profit
                    self.own['volumn'][w] = 0 
                    continue  
        #把數量==0的或從own中刪掉
        self.own = self.own[self.own['volumn']!=0]
        self.own = self.own.reset_index(drop=True)
        
        #成交稅:成交價的0.00002 = 點數直接除250
        #cost = cost + (p_point/250)*abs_buy_num            
        
        position = [act_type,buy_num]
        is_contain_sell = 1
        #持有的口數的最低原始與維持保證金為何
        self.m_acct = self.m_acct - m_margin*abs_buy_num #帳戶中的維持保證金需為
        self.o_acct = self.o_acct - o_margin*abs_buy_num #帳戶中的原始保證金需為
        
        
        
        return cost,profit,position,is_contain_sell        
    
#%%
    def step(self, act, date):
        profit = 0 #獲利
        cost = 0 #成本(買(應該用買的點數*200算還是用原始保證金算?)+手續費+交易稅)
        position = [] #ex:[['TX01', 1], ['TX02', 1]]
        unrealize = 0 #未實現損益
        more_money = 0 #如果低於維持保證金要補多少錢 

        h_fee = 0 #手續費，沒交易就是0
        is_contain_sell= 0 #判斷這次交易有沒有含有賣
        
        if date in self.data.Date.values:
            #算這天的原始保證金和維持保證金為
            for index, row in self.maintain_margin.iterrows():
                start=pd.to_datetime(row['start']).date()
                end=pd.to_datetime(row['end']).date()
                if start <= date <= end:
                    tx_o_margin=row['tx_o_margin']
                    mtx_o_margin=row['mtx_o_margin']
                    tx_m_margin=row['tx_m_margin']
                    mtx_m_margin=row['mtx_m_margin']
                    break
            
            # =============================================================================
            #         執行買賣的action
            # =============================================================================
            for acts in act:
                act_type = acts[0]
                buy_num = acts[1]
    
                #判斷act是大台還是小台 ，近一還是近二
                if act_type == 'TX01':
                    o_margin = tx_o_margin
                    m_margin = tx_m_margin
                    tx_type = 'TX' #之後查資料方便把tx mtx
                    due_mon = 1   # 近一近二先標出來
                    per_mon = 200 #大台200                
    
                elif act_type == 'TX02':
                    o_margin = tx_o_margin
                    m_margin = tx_m_margin
                    tx_type = 'TX'
                    due_mon = 2
                    per_mon = 200               
                    
                elif act_type == 'MTX01':
                    o_margin = mtx_o_margin
                    m_margin = mtx_m_margin
                    tx_type = 'MTX'
                    due_mon = 1
                    per_mon = 50
                    
                elif act_type == 'MTX02':
                    o_margin = mtx_o_margin
                    m_margin = mtx_m_margin
                    tx_type = 'MTX'
                    due_mon = 2
                    per_mon = 50
                  
                else: continue
    
                #有交易就有手續費，手續費50~100都有，取中間值75
#                if buy_num != 0 :
#                    h_fee=75
                sum_vol = self.own[(self.own['type'] == tx_type) & (self.own['due_mon'] == due_mon)]['volumn'].sum() 
                
                #單純買
                if (sum_vol==0) | (sum_vol*buy_num > 0):
                    b_cost,b_position = self.buy(date,buy_num,act_type,o_margin,m_margin,tx_type,due_mon,per_mon)
                    cost += b_cost
                    position.append(b_position)
                #單純賣    
                elif (sum_vol*buy_num < 0) & (abs(sum_vol) >= abs(buy_num)):
                    s_cost,s_profit,s_position,contain_sell = self.sell(date,buy_num,act_type,o_margin,m_margin,tx_type,due_mon,per_mon)
                    #算手續費的
                    is_contain_sell = contain_sell
                    cost += s_cost
                    profit += s_profit
                    position.append(s_position)
                else: #先賣後買   
                    a = -sum_vol
                    b = sum_vol + buy_num
                    #先賣
                    s_cost,s_profit,s_position,contain_sell = self.sell(date,a,act_type,o_margin,m_margin,tx_type,due_mon,per_mon)
                    cost += s_cost
                    profit += s_profit
                    #後買
                    b_cost,b_position = self.buy(date,b,act_type,o_margin,m_margin,tx_type,due_mon,per_mon)
                    cost += b_cost
                    position.append([act_type,s_position[1]+b_position[1]])
                    
            # =============================================================================
            #         for 結束     
            # =============================================================================
            
            #判斷當天是否為到期日，是的話要把近一都平倉掉 
            if date in self.dueDate:
                #結算價為
                p = self.dueDate_price[self.dueDate_price['dueDate']==date]
                due_price = p['price'].iloc[0]            
                
                #把own中大台小台 是近一月且數量>0 都抓出來平掉
                due_own = self.own[(self.own['due_mon']==1) & self.own['volumn']!=0]
                
                #sum_num = due_own['volumn'].sum() #共幾口，算成交稅要用的
                for j,due_row in due_own.iterrows():
                    profit = profit + (due_price - due_row['buy_point'])*due_row['volumn']*due_row['per_price']
                    self.own['volumn'][j] = 0
                    #賣掉-->pool增加
                    self.pool += (due_price-due_row['buy_point'])*due_row['volumn']*due_row['per_price']
                    
                    #持有的口數的最低原始與維持保證金會下降
                    self.m_acct = self.m_acct - m_margin*abs(due_row['volumn']) #帳戶中的維持保證金需為
                    self.o_acct = self.o_acct - o_margin*abs(due_row['volumn']) #帳戶中的原始保證金需為
                    
                #把數量==0的或從own中刪掉
                self.own = self.own[self.own['volumn']!=0]
                self.own = self.own.reset_index(drop=True)
                
                #成交稅:成交價的0.00002 = 點數直接除250
                #cost = cost + (due_price/250)*sum_num
                is_contain_sell = 1               
                
                #同時把own裡面的近二變成近一
                r_due = {2:1}
                self.own = self.own.replace({"due_mon":r_due})
                
            #目前持有的是否跌破維持保證金，若跌破了向env要求補錢(more_money)
            lost=0 #保證金水位上升或下降多少  
            for indexs, rows in self.own.iterrows():  
                close_p = self.data[(self.data['Date']==date) & (self.data['contract_rank']==rows['due_mon']) & (self.data['Symbol']==rows['type'])]['Close'].iloc[0]            
                lost = lost + (close_p - rows['buy_point'])*rows['volumn']*rows['per_price']
            if (self.pool + lost) < self.m_acct:
                more_money = self.o_acct - (self.pool + lost) #若小於維持保證金要補多少錢
            #若有賣，維持 原始保證金*剩餘口數的錢 即可
            if is_contain_sell == 1:         
                if self.pool > self.o_acct: #若現有的pool的錢 > 目前需要的原始保證金的話 
                    return_money = self.pool - self.o_acct
                    self.cash += return_money
                    self.pool = self.o_acct
            
            #算未實現損益   
            for u_index, u_row in self.own.iterrows():  
                close_p = self.data[(self.data['Date']==date) & (self.data['contract_rank']==u_row['due_mon']) & (self.data['Symbol']==u_row['type'])]['Close'].iloc[0]            
                unrealize = unrealize + (close_p - u_row['buy_point'])*u_row['volumn']*u_row['per_price']  
            #算手續費
            cost = cost + h_fee
            
            profit = int(profit)
            cost = int(cost)
            unrealize = int(unrealize)
                
            self.cash += profit - cost
        
        return self.cash, profit, cost, position, unrealize, more_money
#%%    
    #追繳保證金
    def margin_call(self,more_money,date):
        if self.cash >= more_money:
            self.cash -= more_money
            self.pool += more_money
        #若未依規回補保證金，所有的部位自動以市價平倉
        #市價先用open(正常應是12點之後的某一點)
        else:
            date = date + pd.DateOffset(days=1)
            date = date.date()
            for c_index, c_row in self.own.iterrows():  
                open_p = self.data[(self.data['Date']==date) & (self.data['contract_rank']==c_row['due_mon']) & (self.data['Symbol']==c_row['type'])]['Open'].iloc[0]            
                self.pool += (open_p - c_row['buy_point'])*c_row['volumn']*c_row['per_price']
                self.own['volumn'][c_index] = 0
            #把own中數量==0的清掉
            self.own = self.own[self.own['volumn']!=0]
            self.own = self.own.reset_index(drop=True)
            self.m_acct = 0
            self.o_acct = 0
        

#%%
if __name__ == '__main__':
    start_date = '2016/1/19'
    period = 3
    action = list([
            [['TX01',1],['TX02',1]],
            [['TX01',2],['TX02',1]],
            [['TX01',-3],['TX02',-3]],            
            ])   
    
    futures_folder = './futures_data/'
    env = Env(futures_folder)
    env.load()
    cash = 1e+6
    env.reset(cash)
    
    for i in range(period):
        act = action[i]
        s_date = pd.to_datetime(start_date)
        # TODO: [to 非凡] 用 DataOffset 算日期會遇到+1，結果不在資料裡，改成你現在計算日期的方式即可
        date = s_date + pd.DateOffset(days=i)
        date = date.date()
        
        cash, profit, cost, position, unrealize, more_money = env.step(act, date)
        
        #追繳保證金
        if more_money > 0:
            env.margin_call(more_money,date)
        print(cash, profit, cost, position, unrealize, more_money)
    


#%%
#追繳保證金但env不夠時應該全部平倉還是部份
    
    
    