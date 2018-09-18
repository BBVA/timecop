import pandas as pd
import numpy as np
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

from pyramid.arima import auto_arima

import math

import helpers as h


def anomaly_uni_Keras(lista_datos,desv_mse=0):
    temp= pd.DataFrame(lista_datos,columns=['values'])
    data_raw = temp.values.astype("float32")

    scaler = MinMaxScaler(feature_range = (0, 1))
    dataset = scaler.fit_transform(data_raw)

    TRAIN_SIZE = 0.70

    train_size = int(len(dataset) * TRAIN_SIZE)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size-2:len(dataset), :]
    print("Number of entries (training set, test set): " + str((len(train), len(test))))

    # Create test and training sets for one-step-ahead regression.
    window_size = 1
    train_X, train_Y = h.create_dataset(train, window_size)
    test_X, test_Y = h.create_dataset(test, window_size)
    forecast_X, forecast_Y = h.create_dataset(dataset,window_size)
    print forecast_X
    

    print("New training data shape:")
    #print(train_X.shape)
    print(test_X.shape)

    '''
    #############new engine LSTM
    model = Sequential()
    model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.add(Dropout(0))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=800, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)
	'''
    
    model = Sequential()
    model.add(Dense(8,input_dim=window_size,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(train_X, train_Y, epochs=100, batch_size=2, verbose=1)
    

    yhat = model.predict(test_X)
    
    #pyplot.plot(yhat, label='predict')
    #pyplot.plot(test_Y, label='true')
    #pyplot.legend()
    #pyplot.show()


    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

    #pyplot.plot(yhat_inverse, label='predict')
    #pyplot.plot(testY_inverse, label='true')
    #pyplot.legend()
    #pyplot.show()

    lista_puntos = np.arange(train_size, test_size+train_size)
    
    
    testing_data = pd.DataFrame(yhat_inverse,index =lista_puntos,columns=['expected value'])

    rmse = math.sqrt(mean_squared_error(testY_inverse, yhat_inverse))
    mse=mean_squared_error(testY_inverse, yhat_inverse)
    #print('Test RMSE: %.3f' % rmse)
    mae = mean_absolute_error(testY_inverse, yhat_inverse)
    #print ('test mae LSTM = ' +str(mae))

    
    df_aler = pd.DataFrame()
    test=scaler.inverse_transform([test_Y])
    
    df_aler['real_value'] = test[0]
    
    df_aler['expected value'] = yhat_inverse
    df_aler['step'] = np.arange(0, len(yhat_inverse),1)
    df_aler['mae']=mae
    df_aler['mse']=mse
    df_aler['anomaly_score'] = abs(df_aler['expected value'] - df_aler['real_value']) / df_aler['mae']
    
    
    df_aler_ult = df_aler[:5]

    df_aler_ult = df_aler_ult[(df_aler_ult.index==df_aler.index.max())|(df_aler_ult.index==((df_aler.index.max())-1))
                             |(df_aler_ult.index==((df_aler.index.max())-2))|(df_aler_ult.index==((df_aler.index.max())-3))
                             |(df_aler_ult.index==((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = 'FALSE'
    else:
        exists_anom_last_5 = 'TRUE'
    


    df_aler = df_aler[(df_aler['anomaly_score']> 2)]

    max = df_aler['anomaly_score'].max()
    min = df_aler['anomaly_score'].min()
    df_aler['anomaly_score']= ( df_aler['anomaly_score'] - min ) /(max - min)
    
    
    max = df_aler_ult['anomaly_score'].max()
    min = df_aler_ult['anomaly_score'].min()
 
    df_aler_ult['anomaly_score']= ( df_aler_ult['anomaly_score'] - min ) /(max - min)

    pred_scaled =model.predict(forecast_X)


    pred = scaler.inverse_transform(pred_scaled)
    
    #pyplot.plot(testY_inverse, label='true')
    #pyplot.plot(pred, label='fore')
    #pyplot.legend()
    #pyplot.show()

    engine_output={}



    engine_output['rmse'] = str(math.sqrt(mse))
    engine_output['mse'] = int(mse)
    engine_output['mae'] = int(mae)
    #print ('mae' + str(mae))
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='Keras_model'
    df_future= pd.DataFrame(pred[:15],columns=['value'])
    df_future['value']=df_future.value.astype("float64")
    df_future['step']= np.arange( len(lista_datos),len(lista_datos)+15,1)
    print ("future")
    print(df_future)
    engine_output['future'] = df_future.to_dict(orient='record')
    
    # testing_data['excepted value'].astype("float64")
    testing_data['step']=testing_data.index
    testing_data.step.astype("float64")

    print (testing_data.to_dict(orient='record'))
    engine_output['debug'] = testing_data.to_dict(orient='record')

    return (engine_output)
