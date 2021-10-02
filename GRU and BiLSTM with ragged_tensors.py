import os
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
# %matplotlib inline
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

from scipy import stats
# %matplotlib inline

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM,  Bidirectional, TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model

from tqdm import tqdm
import time

from google.colab import drive
drive.mount("/content/gdrive")

filenames = ['N-CMAPSS_DS01-005.h5', 'N-CMAPSS_DS03-012.h5', 'N-CMAPSS_DS04.h5', 'N-CMAPSS_DS05.h5', 'N-CMAPSS_DS06.h5', 'N-CMAPSS_DS07.h5', 'N-CMAPSS_DS08a-009.h5', 'N-CMAPSS_DS08c-008.h5', 'N-CMAPSS_DS08d-010.h5']

# Time tracking, Operation time (min):  0.003
t = time.process_time() 

data = []
y = []

# Load data
for filename in filenames:
    with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))             # W
        X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
        Y_dev = np.array(hdf.get('Y_dev'))             # RUL  
        A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))           # W
        X_s_test = np.array(hdf.get('X_s_test'))       # X_s
        Y_test = np.array(hdf.get('Y_test'))           # RUL  
        A_test = np.array(hdf.get('A_test'))           # Auxiliary
        
        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        A_var = np.array(hdf.get('A_var'))
        
        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20')) 
        A_var = list(np.array(A_var, dtype='U20'))
                          
    W = np.concatenate((W_dev, W_test), axis=0)
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0)
    A = np.concatenate((A_dev, A_test), axis=0)

    W = pd.DataFrame(W, columns= W_var)
    X_s = pd.DataFrame(X_s, columns= X_s_var)
    A = pd.DataFrame(A, columns= A_var)
    Y = pd.DataFrame(Y)
    df= pd.concat([W, X_s], axis=1)

    del W, X_s,  W_dev, X_s_dev, A_dev, Y_dev, W_test, X_s_test,  A_test , Y_test, W_var, X_s_var , A_var

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    Y = pd.DataFrame(scaler.fit_transform(Y))
  
    for u in A['unit'].unique():
        for c in A['cycle'][A.unit==u].unique():
            y.append (Y[0][A.unit==u][A.cycle==c].unique()[0])
            x =np.array([(df[A.unit==u][A.cycle==c].iloc[:,[i]].values.astype('int32')) for i in range(0,18)])
            x = x.reshape(x.shape[1], 18)
            data.append(x)
    del A, Y, df

y = np.array(y)
del df, Y, A, x

X_train, X_test, y_train, y_test = train_test_split(data, y, train_size = 0.8, shuffle = True)
X_train = tf.ragged.constant(X_train)
X_test = tf.ragged.constant(X_test)
del data, y
y_train = np.array(y_train)
y_test = np.array(y_test)


# Create GRU model
def create_gru(units):
    model = Sequential()
    # Input layer
    model.add(Input(shape=(None, 32), ragged=False))
    model.add(GRU(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    # Hidden layer
    model.add(GRU(units=units / 2))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    # Compile model
    model.compile(loss="mse", optimizer="adam")

    return model

# Create BiLSTM model
def create_bilstm(units):
    model = Sequential()
    # Input layer
    model.add(Input(shape=(None, 18), ragged=False))
    model.add(Bidirectional(LSTM(units = units, return_sequences=True)))
    model.add(Dropout(0.2))
    # Hidden layer
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dense(1))
    # Compile model
    model.compile(loss="mse", optimizer="adam")
    
    return model


def fit_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs = 60, validation_data=(X_test, y_test), batch_size = 3)
    
    return history

def save_model(model, history, model_name):
    model.save(model_name)
    model.save_weights((model_name+".h5"))

    with open((model_name+'/trainHistoryDict'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

model_gru = create_gru(128)
history_gru = fit_model(model_gru, X_train, y_train)
save_model(model_gru, history_gru, 'GRU')

model_bilstm = create_bilstm(128)
history_bilstm = fit_model(model_bilstm, X_train, y_train)
save_model(model_bilstm, history_bilstm, 'BiLSTM')