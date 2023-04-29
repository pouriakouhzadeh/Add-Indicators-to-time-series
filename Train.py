#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf


# In[2]:


data=pd.read_csv('BTC_all_indicators.csv')


# In[23]:


data.shape


# In[25]:


data_new = data.drop(['0'], axis = 1)
data_new = data_new.iloc[300000:320000,:]
# training_data_len=math.ceil(len(data_new)*0.99)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data_new)
training_data=scaled_data[:(len(data_new)-180)]
testing_data=scaled_data[(len(data_new)-180):]
print('shape of original data : ' , data.shape)
print('shape of traning data : ' , training_data.shape)
print('shape of testing data : ' , testing_data.shape)


# In[27]:


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

print('shape of X_train : ',X_train.shape)
print('shape of Y_train : ',Y_train.shape)
print('shape of X_test : ',X_test.shape)
print('shape of Y_test : ',Y_test.shape)


# In[28]:


# Conver 3D array to 2D this may take several minutes depending to your dataset
nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))


# In[29]:


d2_test_dataset= np.array(d2_test_dataset)
d2_train_dataset = np.array(d2_train_dataset)


# In[30]:


# select 20 most important featurs
f = 400
X_train_new = SelectKBest(f_classif, k=f).fit_transform(d2_train_dataset, Y_train)


# In[31]:


X_test_new = SelectKBest(f_classif, k=f).fit_transform(d2_test_dataset, Y_test)


# In[32]:


X_train_new=X_train_new.reshape(X_train_new.shape[0],f,1)


# In[33]:


X_train_new.shape


# In[34]:


#Initialize the RNN
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train_new.shape[1], X_train_new.shape[2])))
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


# In[ ]:


model.compile(optimizer ='adam' , loss = 'mean_squared_error') # optimizer ='adam'
history= model.fit(X_train_new, Y_train, epochs = 20, batch_size =None,workers =10,use_multiprocessing =True, validation_split=0.1)  


# In[ ]:


X_test_new=X_test_new.reshape(len(X_test_new),f,1)
prediction=model.predict(X_test_new)


# In[ ]:


df = pd.DataFrame()
df['Y'] =Y_test
df['P'] =prediction.reshape(-1)
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


# In[ ]:


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


# In[ ]:


los


# In[ ]:




