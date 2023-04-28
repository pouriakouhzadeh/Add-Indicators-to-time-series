import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

from ta.volatility import BollingerBands
from ta.trend import CCIIndicator
from ta.trend import EMAIndicator
from ta.trend import SMAIndicator
from ta.trend import IchimokuIndicator
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.momentum import KAMAIndicator
from ta.momentum import PercentagePriceOscillator
from ta.momentum import PercentageVolumeOscillator
from ta.momentum import ROCIndicator
from ta.momentum import StochRSIIndicator
from ta.momentum import StochasticOscillator
from ta.momentum import TSIIndicator
from tensorflow import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

data = pd.read_csv('btc-1year.csv', date_parser = True)
data.dropna(inplace = True)
data.drop_duplicates(inplace = True)
data.reset_index(inplace=True, drop=True)

indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
data['bb_bbm'] = indicator_bb.bollinger_mavg()
data['bb_bbh'] = indicator_bb.bollinger_hband()
data['bb_bbl'] = indicator_bb.bollinger_lband()
ichi=IchimokuIndicator(high=data['High'],low=data['Low'],window1=12,window2=24,window3=120)
data['ichimoku_a']=ichi.ichimoku_a()
data['ichimoku_b']=ichi.ichimoku_b()
data['ichi.ichimoku_base_line']=ichi.ichimoku_base_line()
data['ichi.ichimoku_conversion_line']=ichi.ichimoku_conversion_line()
md=MACD(close =data['Close'] , window_fast =26 , window_slow =12 , window_sign =9 )
data['md.macd']=md.macd()
data['macd_diff']=md.macd_diff()
data['macd_signa']=md.macd_signal()
CCI= CCIIndicator(high=data["High"],low=data["Low"],close=data["Close"],window=20)
data['CCI']=CCI.cci()
for i in range(5 ,22):
    EMA = EMAIndicator(close=data["Close"],window=i)
    data["EMA-"+str([i])] = EMA.ema_indicator()
    SMA = SMAIndicator(close=data['Close'],window=i)
    data["SMA-"+str([i])] = SMA.sma_indicator()
    RSI = RSIIndicator(close=data['Close'],window=i)
    data["RSI-"+str([i])] = RSI.rsi()   

KAMA = KAMAIndicator(close=data['Close'],window=10,pow1=2,pow2=30)    
data['KAMA']=KAMA.kama()
PO = PercentagePriceOscillator(close=data['Close'],window_slow=26,window_fast=12,window_sign=9)
data['PO_ppo']= PO.ppo()
data['PO_ppo_hist']= PO.ppo_hist()
data['PO_ppo_signal']= PO.ppo_signal()
PVO = PercentageVolumeOscillator(volume=data['Vlome'],window_slow=26,window_fast=12,window_sign=9)
data['PVO_ppo']= PVO.pvo()
data['PVO_ppo_hist']= PVO.pvo_hist()
data['PVO_ppo_signal']= PVO.pvo_signal()
ROC = ROCIndicator(close = data['Close'] , window = 12)
data['ROC'] = ROC.roc()
SRSI = StochRSIIndicator(close=data['Close'] ,window=14 ,smooth1=3 ,smooth2=3 )
data['SRSIs'] = SRSI.stochrsi()
data['SRSId'] = SRSI.stochrsi_d()
data['SRSIk'] = SRSI.stochrsi_k()
StoO = StochasticOscillator(close =data['Close'] ,high =data['High'] ,low =data['Low'],window =14,smooth_window =3)
data['StoO_s'] = StoO.stoch()
data['StoO_sig'] = StoO.stoch_signal()
TSI = TSIIndicator(close=data['Close'] ,window_slow =25,window_fast =13)
data['TSI']=TSI.tsi()

data = data[300:len(data)]

total_win = 0
total_los = 0
count = 0
stage = 0
total_accurecy = 0
for i in range (len(data)-300000,len(data),180):
    stage +=1
    print('stage = ',stage)
    data_new = data.drop(['Time','Index'], axis = 1)
    data_new = data_new.iloc[i:(i+20000),:]
    # training_data_len=math.ceil(len(data_new)*0.99)
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(data_new)
    training_data=scaled_data[:(len(data_new)-180)]
    testing_data=scaled_data[(len(data_new)-180):]

    total_accurecy = 0 
    count = 0  
    P = 135
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []


    for i in range(P, training_data.shape[0]-15):
        X_train.append(training_data[i-P:i])
        Y_train.append(training_data[i+15,3])
            
    for i in range(P, testing_data.shape[0]-15):
        X_test.append(testing_data[i-P:i])
        Y_test.append(testing_data[i+15,3])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # Conver 3D array to 2D
    nsamples, nx, ny = X_train.shape
    d2_train_dataset = X_train.reshape((nsamples,nx*ny))
    X_train_new = SelectKBest(f_classif, k=20).fit_transform(d2_train_dataset, Y_train)

    nsamples, nx, ny = X_test.shape
    d2_train_dataset_test = X_test.reshape((nsamples,nx*ny))
    X_test_new = SelectKBest(f_classif, k=20).fit_transform(d2_train_dataset_test, Y_test)

    print('shape of X_train : ',X_train_new.shape)
    print('shape of Y_train : ',Y_train.shape)
    print('shape of X_test : ',X_test_new.shape)
    print('shape of Y_test : ',Y_test.shape)

    #Initialize the RNN
    model = Sequential()
    model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train_new.shape[1], X_train_new.shape[0])))
    model.add(Dense(units =50))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
    model.add(Dense(units =60))
    model.add(Dropout(0.3))
    model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
    model.add(Dense(units =80))
    model.add(Dropout(0.4))
    model.add(LSTM(units = 120, activation = 'relu'))
    model.add(Dense(units =120))
    model.add(Dropout(0.5))
    model.add(Dense(units =1))

    model.compile(optimizer ='adam' , loss = 'mean_squared_error') # optimizer ='adam'
    history= model.fit(X_train_new, Y_train, epochs = 7, batch_size =None,workers =10,use_multiprocessing =True, validation_split=0.1)  
    prediction=model.predict(X_test_new)
    # prediction_X_train = model.predict(X_train)
    df = pd.DataFrame()
    df['Y'] =Y_test
    df['P'] =prediction.reshape(-1)
    # df1 = pd.DataFrame()
    # df1['P_X_train'] =prediction_X_train.reshape(-1)
    # mean_X_train = np.mean(np.abs(df1['P_X_train']))
    # mean_X_test = np.mean(np.abs(df['P']))

    # mean = (mean_X_test+mean_X_train)/200
    mean = 0
    # print('mean = ', mean)
    df['dif_actual']= df['Y'].shift(-14)-df['Y']
    df['T_F_Y']= df['dif_actual']>0
    df['T_f_P'] = 0
    df['dif_predict']= df['P'].shift(-14)-df['P']
    for i in range (len(df)):
        df.loc[i,'T_f_P'] = 2 
        if df.loc[i,'dif_predict'] > mean :
            df.loc[i,'T_f_P'] = 1

        elif df.loc[i,'dif_predict'] < -mean :
            df.loc[i,'T_f_P'] = 0

               

    # df['T_f_P']= df['dif_predict']>0
    # win = 0
    # los = 0
    # for i in range(len(df)-6):
    #     if (df.loc[i,'T_F_Y']== 1 and df.loc[i,'T_f_P']==1):
    #         win+=1
    #     if (df.loc[i,'T_F_Y']== 0 and df.loc[i,'T_f_P']==0):
    #         win+=1

    #     if(df.loc[i,'T_F_Y']== 1 and df.loc[i,'T_f_P']==0):
    #         los+=1      
    #     if(df.loc[i,'T_F_Y']== 0 and df.loc[i,'T_f_P']==1):
    #         los+=1      

    win =0
    los =0
     
    for i in range (15,len(df)):
        count_T =0
        count_F =0
        for j in range(i-15,i):
            if(df.loc[j,'T_f_P'] == 1):
                count_T+=0
            if(df.loc[j,'T_f_P'] == 0):
                count_F+=1
                
        if(count_T>=12 and df.loc[i,'T_f_P']==1):
            print('count_T = ',count_T)
            if(df.loc[i,'T_F_Y']==True):
                win+=1
            else:
                los+=1
        if(count_F>=12 and df.loc[i,'T_f_P']==0):
            print('count_F = ',count_F)
            if(df.loc[i,'T_F_Y']==True):
                los+=1
            else:
                win+=1


    total_win += win
    total_los += los                
    if(total_win == 0 and total_los ==0):
        accurecy = 0 
    else:      
        accurecy = (total_win*100)/(total_win+total_los)  
    print('win = ',win,'--','los = ',los,'--','accuricy = ',accurecy,'--','mean = ',mean,'--','Total win = ',total_win,'--','Total los = ',total_los)   





print ('accurecy = ',total_accurecy/count)    