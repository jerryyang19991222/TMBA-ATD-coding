#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import talib as ta
from talib import abstract


# In[15]:


def KDJ (traindata,fund,feePaid):  
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算移動平均線
    traindata['lma'] = ta.SMA(traindata['Close'], timeperiod=144)
    traindata = traindata.dropna(subset=['lma'])
    traindata['sma'] = ta.SMA(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['sma'])
    #計算KDJ
    n=9
    traindata['L_n'] = traindata['Low'].rolling(window=n).min()
    traindata['H_n'] = traindata['High'].rolling(window=n).max()
    traindata['RSV'] = 100 * (traindata['Close'] - traindata['L_n']) / (traindata['H_n'] - traindata['L_n'])
    traindata['K'] = traindata['RSV'].ewm(span=3, adjust=False).mean()
    traindata['D'] = traindata['K'].ewm(span=3, adjust=False).mean()
    traindata['J'] = 3 * traindata['K'] - 2 * traindata['D']
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
    #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['J'][i]<0
        entryCondition =traindata['lma'][i]<traindata['sma'][i]
        ## 出場邏輯
        exitShort = traindata['J'][i]>100
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][t] -traindata['atr'][i]*1
            # 加碼條件
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong and entryCondition:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list))*1.1)/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))*2
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)

    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity 


# In[19]:


def BBand(traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'], traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算KDJ
    n=9
    traindata['L_n'] = traindata['Low'].rolling(window=n).min()
    traindata['H_n'] = traindata['High'].rolling(window=n).max()
    traindata['RSV'] = 100 * (traindata['Close'] - traindata['L_n']) / (traindata['H_n'] - traindata['L_n'])
    traindata['K'] = traindata['RSV'].ewm(span=3, adjust=False).mean()
    traindata['D'] = traindata['K'].ewm(span=3, adjust=False).mean()
    traindata['J'] = 3 * traindata['K'] - 2 * traindata['D']
    #計算RSI
    traindata['rsistrat'] = ta.RSI(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['rsistrat'])
    #計算BBand
    traindata['Upper BBand'], traindata['Middle BBand'], traindata['Lower BBand'] = ta.BBANDS(traindata['Close'],timeperiod=20,nbdevup=2,nbdevdn=2,matype=0)
    traindata['BBandw']=traindata['Upper BBand']-traindata['Lower BBand']
    #計算SMA斜率
    traindata['sma'] = ta.SMA(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['sma'])
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    traindata['SMA_5day_slope'] = traindata['sma'].rolling(window=5).apply(calculate_slope, raw=False)
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['Low'][i]<traindata['Lower BBand'][i] and traindata['rsistrat'][i]<50
        entryCondition =traindata['SMA_5day_slope'][i]>0 
        ## 出場邏輯
        exitShort = traindata['J'][i]>100
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong and entryCondition:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list))*1.5)/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity 


# In[22]:


def twoMA (traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算移動平均線
    traindata['lma'] = ta.SMA(traindata['Close'], timeperiod=144)
    traindata = traindata.dropna(subset=['lma'])
    traindata['sma'] = ta.SMA(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['sma'])
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['lma'][i]<traindata['sma'][i]
        ## 出場邏輯
        exitShort =  traindata['lma'][i]<traindata['sma'][i]
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list))*0.8)/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity


# In[25]:


def Donchian (traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算SMA斜率
    traindata['sma'] = ta.SMA(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['sma'])
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    traindata['SMA_5day_slope'] = traindata['sma'].rolling(window=5).apply(calculate_slope, raw=False)
    #計算董銓通道
    traindata['Upper Band'] = traindata['Close'].rolling(window=5).max()
    traindata['Lower Band'] = traindata['Close'].rolling(window=5).min()
    traindata['Middle Band'] = (traindata['Upper Band'] + traindata['Lower Band']) / 2
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['High'][i]=traindata['Upper Band'][i]
        entryCondition =traindata['SMA_5day_slope'][i]>0 
        ## 出場邏輯
        exitShort = traindata['SMA_5day_slope'][i]<0 
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong and entryCondition:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list))*0.8)/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity


# In[28]:


def ATRmeanreversion(traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'], traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算RSI
    traindata['rsistrat'] = ta.RSI(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['rsistrat'])
    #ATR通道
    traindata['Middle ATRBand']= ta.SMA(traindata['Close'], timeperiod=20)
    traindata['Upper ATRBand'] =traindata['Middle ATRBand']+traindata['atr']*2
    traindata['Lower ATRBand'] =traindata['Middle ATRBand']-traindata['atr']*2
    #計算SMA斜率
    traindata['sma'] = ta.SMA(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['sma'])
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    traindata['SMA_5day_slope'] = traindata['sma'].rolling(window=5).apply(calculate_slope, raw=False)
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['Low'][i]<traindata['Lower ATRBand'][i] and traindata['rsistrat'][i]<50
        entryCondition =traindata['SMA_5day_slope'][i]>0 
        ## 出場邏輯
        exitShort = traindata['SMA_5day_slope'][i]<0 
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong and entryCondition:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list))*1.2)/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity


# In[32]:


def ADX (traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算移動平均線
    traindata['lma'] = ta.SMA(traindata['Close'], timeperiod=144)
    traindata = traindata.dropna(subset=['lma'])
    traindata['sma'] = ta.SMA(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['sma'])
    #計算RSI
    traindata['rsistrat'] = ta.RSI(traindata['Close'], timeperiod=55)
    traindata = traindata.dropna(subset=['rsistrat'])
    #計算DMA
    traindata['+DI'] = ta.PLUS_DI(traindata['High'], traindata['Low'], traindata['Close'], timeperiod=14)
    traindata['-DI'] = ta.MINUS_DI(traindata['High'], traindata['Low'], traindata['Close'], timeperiod=14)
    traindata['ADX'] = ta.ADX(traindata['High'], traindata['Low'],traindata['Close'], timeperiod=14)
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['ADX'][i]>20
        entryCondition =traindata['lma'][i]<traindata['sma'][i]
        ## 出場邏輯
        exitShort =traindata['ADX'][i]>100
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong and entryCondition:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list)))/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity


# In[34]:


def CCI (traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算CCI
    traindata['CCI'] = ta.CCI(traindata['High'], traindata['Low'], traindata['Close'], timeperiod=6)
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    traindata['CCI_3day_slope'] = traindata['CCI'].rolling(window=3).apply(calculate_slope, raw=False)
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['CCI'][i]<-100 and traindata['CCI_3day_slope'][i]>0
        ## 出場邏輯
        exitShort =traindata['CCI'][i]>100 and traindata['CCI_3day_slope'][i]<0
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list))*2)/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity


# In[39]:


def alligator(traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    # 上顎线（Jaw）：13周期
    traindata['Jaw'] = ta.SMA((traindata['High']-traindata['Low'])/2, timeperiod=13)
    # 牙齿线（Teeth）：8周期
    traindata['Teeth'] = ta.SMA((traindata['High']-traindata['Low'])/2, timeperiod=8)
    # 唇线（Lips）：5周期
    traindata['Lips'] = ta.SMA((traindata['High']-traindata['Low'])/2, timeperiod=5)
    #計算DMA
    traindata['+DI'] = ta.PLUS_DI(traindata['High'], traindata['Low'], traindata['Close'], timeperiod=14)
    traindata['-DI'] = ta.MINUS_DI(traindata['High'], traindata['Low'], traindata['Close'], timeperiod=14)
    traindata['ADX'] = ta.ADX(traindata['High'], traindata['Low'],traindata['Close'], timeperiod=14)
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['Lips'][i]>traindata['Teeth'][i] and traindata['Teeth'][i]>traindata['Jaw'][i] and traindata['ADX'][i]>20
        ## 出場邏輯
        exitShort =(traindata['Lips'][i]<traindata['Teeth'][i] and traindata['Teeth'][i]<traindata['Jaw'][i]) or traindata['ADX'][i]<20
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list)*0.5))/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity


# In[43]:


def MACD(traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #計算MACD
    macd, macdsignal, macdhist = ta.MACD(traindata['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    traindata['MACD'] = macd
    traindata['MACD Signal'] = macdsignal
    traindata['MACD Histogram'] = macdhist
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata)):
        #回測期間最後一天就跳出這個迴圈
        if i == len(traindata)-1:
            break
        ## 進場邏輯
        entryLong = traindata['MACD Histogram'][i]>0 and traindata['MACD Histogram'][i-1]<0 
        ## 出場邏輯
        exitShort =traindata['MACD Histogram'][i]<0 and traindata['MACD Histogram'][i-1]>0 
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][i-1] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list)*0.5))/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity    


# In[47]:


def Vegas(traindata,fund,feePaid):
    #計算ATR
    traindata['atr'] = ta.ATR(traindata['High'],traindata['Low'], traindata['Close'], timeperiod=20)
    traindata = traindata.dropna(subset=['atr'])
    #upper
    traindata['ema576'] = ta.EMA(traindata['Close'], timeperiod=576)
    traindata = traindata.dropna(subset=['ema576'])
    traindata['ema676'] = ta.EMA(traindata['Close'], timeperiod=676)
    traindata = traindata.dropna(subset=['ema676'])
    #lower
    traindata['ema144'] = ta.EMA(traindata['Close'], timeperiod=144)
    traindata = traindata.dropna(subset=['ema144'])
    traindata['ema169'] = ta.EMA(traindata['Close'], timeperiod=169)
    traindata = traindata.dropna(subset=['ema169'])
    #middle
    traindata['ema12'] = ta.EMA(traindata['Close'], timeperiod=12)
    traindata = traindata.dropna(subset=['ema12'])
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    traindata['EMA576_5day_slope'] = traindata['ema576'].rolling(window=5).apply(calculate_slope, raw=False)
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    traindata['EMA676_5day_slope'] = traindata['ema676'].rolling(window=5).apply(calculate_slope, raw=False)
    #計算KDJ
    n=9
    traindata['L_n'] = traindata['Low'].rolling(window=n).min()
    traindata['H_n'] = traindata['High'].rolling(window=n).max()
    traindata['RSV'] = 100 * (traindata['Close'] - traindata['L_n']) / (traindata['H_n'] - traindata['L_n'])
    traindata['K'] = traindata['RSV'].ewm(span=3, adjust=False).mean()
    traindata['D'] = traindata['K'].ewm(span=3, adjust=False).mean()
    traindata['J'] = 3 * traindata['K'] - 2 * traindata['D']
    #參數設定
    BS = None
    Position = 0
    buy_price = 0
    sell_price = 0
    add_price = 0
    buy = []
    sell = []
    sellshort = []
    buytocover = []
    profit_list = [0]
    profit_fee_list = [0]
    profit_fee_list_realized = []
    cost_list = []
    rets = []
    time_arr = np.array(traindata.index)
    sellcount=0
    for i in range(len(traindata) - 1):
        # Define entry and exit conditions
        entryLong = traindata['ema12'][i] > traindata['ema144'][i] and traindata['ema12'][i] > traindata['ema169'][i]
        entryConditiona = traindata['Close'][i] > traindata['ema144'][i] and traindata['Close'][i] > traindata['ema169'][i]
        entryConditionb = traindata['EMA576_5day_slope'][i] > 0 and traindata['EMA676_5day_slope'][i] > 0
        exitShort = traindata['J'][i]>100
        exitConditiona = traindata['Close'][i] < traindata['ema144'][i] and traindata['Close'][i] < traindata['ema169'][i]
        exitConditionb = traindata['EMA576_5day_slope'][i] < 0 and traindata['EMA676_5day_slope'][i] < 0
        if BS == 'B':
            # 停利停損條件
            stopLoss = traindata['Close'][i] <= traindata['Close'][t] -traindata['atr'][i]*1
            add_condition = traindata['Close'][i] > buy_price
        if (BS == None) & (Position == 0):
            profit_list.append(0)
            profit_fee_list.append(0)
            #確認進場&相關設定
            if entryLong and entryConditiona and entryConditionb:
                #更改狀態至做多
                BS = 'B'
                a=round(((1000000+np.sum(profit_fee_list))*2)/(traindata['atr'][i]*traindata['Close'][i]),0)
                Position = a
                #紀錄進場時間
                t = i+1
                buy_price = traindata['Close'][i+1]*a
                cost_list.append(buy_price)
                buy.append(t)
        elif BS == 'B':
            profit = (traindata['Close'][i+1] - traindata['Close'][i])* Position
            profit_list.append(profit)
            sellcount+=1
            #近場條件達成，計算未實現損益-交易成本
            if exitShort  or i == len(traindata)-2 or stopLoss :
                pl_round = ((Position*traindata['Close'][i+1]) - np.sum(cost_list[-2:]))*2
                profit_fee = profit - feePaid*2
                profit_fee_list.append(profit_fee)
                sell.append(i+1)
                # Realized PnL
                profit_fee_realized = pl_round - feePaid*2*Position
                profit_fee_list_realized.append(profit_fee_realized)
                rets.append(profit_fee_realized/(traindata['Close'][t]))
                #重置交易狀態
                BS = None
                #重置部位數量
                Position = 0
                #重置加碼參考價
                buy_price = 0
                #重置加碼成本
                add_price = 0
                buy.append(t)
                sellcount=0
            elif (Position <=10000)& add_condition:
                #更改部位數量
                buy_price = traindata['Close'][i+1]*1000
                #將第一次進場成本紀錄在這個list中
                cost_list.append(buy_price)
                Position += 1000
                profit_fee = profit
                profit_fee_list.append(profit_fee)
                buy.append(t)
            #出場條件未達成，計算未實現損益
            else:
                profit_fee = profit
                profit_fee_list.append(profit_fee)
    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=traindata.index)
    equity['equity'] = equity['profitfee'] + fund
    equity['strategy_ret'] = equity['equity'].pct_change()
    equity['cum_strategy_ret'] = equity['strategy_ret'].cumsum()
    equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
    equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
    return equity    


# In[48]:


import yfinance as yf
daydata =yf.download('00708L.TW', start='2010-01-01', end='2024-08-30')
traindata = daydata['2010-01-01':'2024-08-30']
Vegas(traindata,1000000,200)


# In[ ]:




