# -*- coding: utf-8 -*-

import collections
import time
import os
import datetime as dt
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import warnings
from numpy import newaxis
from keras import Input, layers
from keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Reshape, Conv1D, MaxPool1D
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adagrad, RMSprop, Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler,Normalizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



############    global variable  #######################
start = time.time()
filetype=['netrad.xlsx','precip.xlsx','temperature.xlsx','wind.xlsx']
datatype=['netrad','precip','temperature','wind']
data = collections.defaultdict(list)
data1 = collections.defaultdict(list)
time_split = [9131,8766,8401,7305,5479,3653]
##2015,2014,2013,2010,2005
day = 7
filename = r'./YW.xlsx'
preprocess = {'pca':[1,4,1,5],'kmeans':[4,3,3,3],'mean':[1,1,1,1]}   
col = preprocess['pca']



###########   dataset   ################################
for i,f in enumerate(filetype):
    filepath=r'./PCA/%s' %f
    scaler=MinMaxScaler()
    Rdata=np.array(pd.read_excel(filepath))[:8401,:col[i]]
    for k in range(len(Rdata)-day):
        data[datatype[i]]=np.append(data[datatype[i]],Rdata[k:k+day,:])
    data[datatype[i]]=scaler.fit_transform(np.reshape(data[datatype[i]],(len(Rdata)-day,len(Rdata[0])*day)))
x=np.concatenate((data[datatype[0]],data[datatype[1]],data[datatype[2]],data[datatype[3]]
                ),axis=1)
x=x.reshape(x.shape[0],1,x.shape[1])
dataframe = pd.read_excel(filename)
Ry = dataframe.values[:,1][day:8401]
scaler1 = MinMaxScaler()
y = scaler1.fit_transform(np.array(Ry).reshape(-1,1))
timesplit = time_split[4]
train_x,test_x = x[:timesplit,:],x[timesplit:,:]
train_y,test_y = y[:timesplit,:],y[timesplit:,:]



######### batchsize epoch neurons  ####################
batch_size = 32
epochs = 100
neuron = [sum(preprocess['pca'])*day,100,100,1]


################  LSTM model  ##########################
model=Sequential()
model.add(LSTM(neuron[1],activation='relu',input_shape=(1,neuron[0]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(neuron[2],activation='relu',return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(neuron[3],activation='linear'))


############### training  #############################
lr = 8e-4
algorithm = Adam(lr=lr)
model.compile(optimizer=algorithm,loss='mse')       
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5,monitor='val_loss')]      
history = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,shuffle=False,\
          callbacks=callbacks,validation_data=(test_x,test_y))#validation_split=0.1
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()
print('Model Training Completed')
print('Finish Time:%d s' %(int(time.time()-start)))




################# predict  #########################
Y=scaler1.inverse_transform(np.array(model.predict(x)).reshape(-1,1))
predict_y=Y[timesplit:]
true_y=Ry[timesplit:]



###################   save image  #####################
save_path=os.path.join('LSTM %s.png' %dt.datetime.now().strftime('%Y%m%d%H%M%S'))
plt.figure(figsize=(15,5))
plt.plot(true_y,color='r',label='true')
plt.plot(predict_y,color='b',label='lstm')
plt.legend()
plt.savefig(save_path)
plt.show()


################   performance metrics   ###############
print('lr:%f'%lr)
sum_a=sum([(true_y[i]-predict_y[i])**2 for i in range(len(true_y))])
sum_b=sum([(true_y[i]-np.mean(true_y))**2 for i in range(len(true_y))])
NSE=1-sum_a/sum_b
print('LSTM NSE Value:%.4f' %NSE) 
print('R2 value: %.4f' %r2_score(true_y,predict_y))
print('RMSE: %.4f' %(np.sqrt(mean_squared_error(true_y,predict_y))))
print('MAE: %.4f' %(mean_absolute_error(true_y,predict_y)))
  



###########   variable  importance ############### 

######################  R  ###########################
data_r = collections.defaultdict(list)
filetype_r = ['netrad_.xlsx','precip.xlsx','temperature.xlsx','wind.xlsx']
for i,f in enumerate(filetype_r):
    filepath = r'./PCA/%s' %f
    scaler = MinMaxScaler()
    Rdata = scaler.fit_transform(np.array(pd.read_excel(filepath))[:8401,:col[i]])
    for k in range(len(Rdata)-day):
        data_r[datatype[i]] = np.append(data_r[datatype[i]],Rdata[k:k+day,:])
    data_r[datatype[i]] = np.reshape(data_r[datatype[i]],(len(Rdata)-day,len(Rdata[0])*day))

x_r = np.concatenate((data_r[datatype[0]],data_r[datatype[1]],data_r[datatype[2]],data_r[datatype[3]]
                ),axis=1)
x_r=x_r.reshape(x_r.shape[0],1,x_r.shape[1])
Y_r = Y - scaler1.inverse_transform(np.array(model.predict(x_r)).reshape(-1,1))


######################  P  ###########################
data_p = collections.defaultdict(list)
filetype_p = ['netrad.xlsx','precip_.xlsx','temperature.xlsx','wind.xlsx']
for i,f in enumerate(filetype_p):
    filepath = r'./PCA/%s' %f
    scaler = MinMaxScaler()
    Rdata = scaler.fit_transform(np.array(pd.read_excel(filepath))[:8401,:col[i]])
    for k in range(len(Rdata)-day):
        data_p[datatype[i]] = np.append(data_p[datatype[i]],Rdata[k:k+day,:])
    data_p[datatype[i]] = np.reshape(data_p[datatype[i]],(len(Rdata)-day,len(Rdata[0])*day))

x_p = np.concatenate((data_p[datatype[0]],data_p[datatype[1]],data_p[datatype[2]],data_p[datatype[3]]
                ),axis=1)
x_p=x_p.reshape(x_p.shape[0],1,x_p.shape[1])
Y_p =  Y - scaler1.inverse_transform(np.array(model.predict(x_p)).reshape(-1,1))


######################  T  ###########################
data_t = collections.defaultdict(list)
filetype_t = ['netrad.xlsx','precip.xlsx','temperature_.xlsx','wind.xlsx']
for i,f in enumerate(filetype_t):
    filepath = r'./PCA/%s' %f
    scaler = MinMaxScaler()
    Rdata = scaler.fit_transform(np.array(pd.read_excel(filepath))[:8401,:col[i]])
    for k in range(len(Rdata)-day):
        data_t[datatype[i]] = np.append(data_t[datatype[i]],Rdata[k:k+day,:])
    data_t[datatype[i]] = np.reshape(data_t[datatype[i]],(len(Rdata)-day,len(Rdata[0])*day))

x_t = np.concatenate((data_t[datatype[0]],data_t[datatype[1]],data_t[datatype[2]],data_t[datatype[3]]
                ),axis=1)
x_t=x_t.reshape(x_t.shape[0],1,x_t.shape[1])
Y_t =  Y - scaler1.inverse_transform(np.array(model.predict(x_t)).reshape(-1,1))


######################  W  ###########################
data_w = collections.defaultdict(list)
filetype_w = ['netrad.xlsx','precip.xlsx','temperature.xlsx','wind_.xlsx']
for i,f in enumerate(filetype_w):
    filepath = r'./PCA/%s' %f
    scaler = MinMaxScaler()
    Rdata = scaler.fit_transform(np.array(pd.read_excel(filepath))[:8401,:col[i]])
    for k in range(len(Rdata)-day):
        data_w[datatype[i]] = np.append(data_w[datatype[i]],Rdata[k:k+day,:])
    data_w[datatype[i]] = np.reshape(data_w[datatype[i]],(len(Rdata)-day,len(Rdata[0])*day))

x_w = np.concatenate((data_w[datatype[0]],data_w[datatype[1]],data_w[datatype[2]],data_w[datatype[3]]
                ),axis=1)
x_w=x_w.reshape(x_w.shape[0],1,x_w.shape[1])
Y_w = Y - scaler1.inverse_transform(np.array(model.predict(x_w)).reshape(-1,1))

