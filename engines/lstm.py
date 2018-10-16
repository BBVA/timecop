
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
import math
import helpers as h
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization






def anomaly_uni_LSTM(lista_datos,num_fut,desv_mse=0):
    temp= pd.DataFrame(lista_datos,columns=['values'])
    data_raw = temp.values.astype("float32")

    scaler = MinMaxScaler(feature_range = (0, 1))
    dataset = scaler.fit_transform(data_raw)
    print ("dataset")
    print (dataset)


    print(data_raw)
    TRAIN_SIZE = 0.70

    train_size = int(len(dataset) * TRAIN_SIZE)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size-2:len(dataset), :]

    # Create test and training sets for one-step-ahead regression.
    window_size = 1
    train_X, train_Y = h.create_dataset(train, window_size)
    test_X, test_Y = h.create_dataset(test, window_size)
    forecast_X, forecast_Y = h.create_dataset(dataset,window_size)

    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    forecast_X = np.reshape(forecast_X, (forecast_X.shape[0], 1, forecast_X.shape[1]))


    print ("train _X") 
    print(train_X)
    print ("train _Y") 
    print(train_Y)
    
    print ("test_X")
    print (test_X)
    print ("test_Y")
    print (test_Y)


    print ("forecast X")
    print (forecast_X)
    #############new engine LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=300, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False)

    yhat = model.predict(test_X)
    
    print ("estoy")
    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

    print (len(test_X))
    print (len(test_Y))
    
    
    
    
    
    #############prediccion LSTM
    salida = forecast_X
    for contador in range(0,num_fut):
        print ("entra " + str(contador))
        preds_one_step = model.predict(salida)   
        # supongo que esta en tamano correcto
        print ("prediccion total" + str(len(preds_one_step)) + "forecast X" + str(len(forecast_X)) + str(len(dataset)) )
        #print (preds_one_step)
        print ( "punto predicho" + str(preds_one_step[-1] ))
        salida = np.append(salida,preds_one_step[-1])
        salida = salida.reshape((salida.shape[0],1,1))
    
    
    print ("forecast complete"+ str(len(salida)))
    print (salida)
    
    lista_puntos = np.arange(train_size, train_size + test_size,1)
    
    print (lista_puntos)
    testing_data = pd.DataFrame(yhat_inverse,index =lista_puntos,columns=['expected value'])

    rmse = math.sqrt(mean_squared_error(testY_inverse, yhat_inverse))
    mse=mean_squared_error(testY_inverse, yhat_inverse)
    mae = mean_absolute_error(testY_inverse, yhat_inverse)



   
    print ("pasa")
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

    print (salida)
    pred = scaler.inverse_transform(salida.reshape(-1, 1))
    
    engine_output={}


    engine_output['rmse'] = str(math.sqrt(mse))
    engine_output['mse'] = int(mse)
    engine_output['mae'] = int(mae)
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='LSTM'
    df_future= pd.DataFrame(pred[len(pred)-num_fut:],columns=['value'])
    df_future['value']=df_future.value.astype("float64")
    df_future['step']= np.arange( len(lista_datos),len(lista_datos)+num_fut,1)
    engine_output['future'] = df_future.to_dict(orient='record')
    print ("llegamos hasta aqui")
    #testing_data['excepted value'].astype("float64")
    testing_data['step']=testing_data.index
    #testing_data.step.astype("float64")
    print ("llegamos hasta aqui2")
    engine_output['debug'] = testing_data.to_dict(orient='record')

    return (engine_output)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg






def anomaly_LSTM(list_var,num_fut,desv_mse=0):
    df_var = pd.DataFrame()
    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]
    print df_var.head(3)

    normalized_df = (df_var-df_var.min())/(df_var.max()-df_var.min())
    print normalized_df.head(3)


    values = normalized_df.values

    TRAIN_SIZE = 0.70
    train_size = int(len(values) * TRAIN_SIZE)
    test_size = len(values) - train_size
    train, test = values[0:train_size, :], values[train_size:len(values), :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    model = Sequential()

    model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(30,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(30))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mae', optimizer='adam')
    
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), shuffle=False)

    yhat = model.predict(test_X)


    ###################################Desnormalizacion#############################################
    y_hat_df = pd.DataFrame()
    y_hat_df['yhat'] = yhat[:,0]

    test_y_df = pd.DataFrame()
    test_y_df['yhat'] = test_y

    #nos quedamos con la columna inicial la cual predecimos para desnormalizar
    ult = df_var[[df_var.columns[-1]]]
    ult['yhat'] = ult[df_var.columns[-1]]
    ult.drop(columns=[df_var.columns[-1]],inplace=True)

    #pyplot.plot(normalized_df[normalized_df.columns[-1]], label='real')
    #pyplot.plot(y_hat_df, label='pred')
    #pyplot.legend()
    #pyplot.show()

    op1= (ult.max()-ult.min())

    desnormalize_y_hat_df = (y_hat_df * op1)+ult.min()

    desnormalize_test_y_df = (test_y_df * op1)+ult.min()

    #pyplot.plot(desnormalize_test_y_df, label='real')
    #pyplot.plot(desnormalize_y_hat_df, label='pred')
    #pyplot.legend()
    #pyplot.show()


    test_y_list = desnormalize_test_y_df['yhat'].tolist()
    yhat_list = desnormalize_y_hat_df['yhat'].tolist()

    ################################### Fin Desnormalizacion#############################################

    mse = (mean_squared_error(test_y_list, yhat_list))
    rmse = np.sqrt(mse)
    df_aler = pd.DataFrame()
    print 'mse', mse
    print 'rmse', rmse

    print ('yhat_list',len(yhat_list))
    print ('test_y_list',len(test_y_list))
    print ('values',len(values))
    print ('train_size',train_size)
    print ('test_size',test_size)

    lista_puntos = np.arange(train_size, train_size + test_size,1)
    testing_data = pd.DataFrame(yhat_list,index =lista_puntos,columns=['expected value'])



    df_aler['real_value'] = test_y_list
    df_aler['expected_value'] = yhat_list

    df_aler['mse'] = mse
    df_aler['puntos'] = df_aler.index
    df_aler['puntos'] = df_aler['puntos'] + train_size
    df_aler.set_index('puntos',inplace=True)


    df_aler['rmse'] = rmse
    mae = mean_absolute_error(yhat_list, test_y_list)
    df_aler['mae'] = mean_absolute_error(yhat_list, test_y_list)


    df_aler['anomaly_score'] = abs(df_aler['expected_value']-df_aler['real_value'])/df_aler['mae']
    print df_aler
    df_aler_ult = df_aler[:5]
    df_aler = df_aler[(df_aler['anomaly_score']> 2)]
    max_anom = df_aler['anomaly_score'].max()
    min_anom = df_aler['anomaly_score'].min()

    df_aler['anomaly_score']= ( df_aler['anomaly_score'] - min_anom ) /(max_anom - min_anom)

    print ('Anomaly')

    print df_aler
    df_aler_ult = df_aler[:5]

    df_aler_ult = df_aler_ult[(df_aler_ult.index==df_aler.index.max())|(df_aler_ult.index==((df_aler.index.max())-1))
                         |(df_aler_ult.index==((df_aler.index.max())-2))|(df_aler_ult.index==((df_aler.index.max())-3))
                         |(df_aler_ult.index==((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = 'FALSE'
    else:
        exists_anom_last_5 = 'TRUE'
    max_ult = df_aler_ult['anomaly_score'].max()
    min_ult = df_aler_ult['anomaly_score'].min()

    df_aler_ult['anomaly_score']= ( df_aler_ult['anomaly_score'] - min_ult ) /(max_ult - min_ult)
    df_aler_ult = df_aler_ult.fillna(0)

    ################################### Forecast #############################################



    test1_X, test1_y = values[:, :-1], values[:, -1]
    test1_X = test1_X.reshape((test1_X.shape[0], 1, test1_X.shape[1]))

    model = Sequential()

    model.add(LSTM(30, input_shape=(test1_X.shape[1], test1_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(30,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(30))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    history = model.fit(test1_X, test1_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

    num_fut=num_fut
    len_fore = len(test1_X) - num_fut
    fore = test1_X[len_fore:]
    yhat_fore = model.predict(fore)



    ###################################Desnormalizacion#############################################
    y_hat_df_fore = pd.DataFrame()
    y_hat_df_fore['yhat'] = yhat_fore[:,0]


    op1= (ult.max()-ult.min())

    desnormalize_y_hat_fore = (y_hat_df_fore * op1)+ult.min()



    #pyplot.plot(desnormalize_y_hat_fore, label='pred')
    #pyplot.legend()
    #pyplot.show()

    yhat_fore_list = desnormalize_y_hat_fore['yhat'].tolist()



    lista_result = np.arange(len(test1_X), (len(test1_X)+num_fut),1)
    df_result_forecast = pd.DataFrame({'puntos':lista_result, 'valores':yhat_fore_list})
    df_result_forecast.set_index('puntos',inplace=True)
    df_result_forecast['valores']=df_result_forecast['valores'].astype(str)
    df_result_forecast['step'] = df_result_forecast.index

    engine_output={}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.to_dict(orient='record')
    engine_output['past']=df_aler.to_dict(orient='record')
    engine_output['engine']='LSTM'
    engine_output['future']= df_result_forecast.to_dict(orient='record')

    testing_data['step']=testing_data.index
    engine_output['debug'] = testing_data.to_dict(orient='record')

    return (engine_output)





#list_var=[5938,6925,4353,6855,6216,7061,4417,7505,6778,8169,5290,7710,6213,7952,5476,7990,7747,8908,5369,7794,7543,8533,5564,6997,6678,7730,4790,6027,5877,7413,4459,5966,5305,6238,3689,5146,4799,6044,3695,4795,4147,5254,2679,3428,3635,5391,3142,3986,3807,4732,2513,3834,3133,4405]

#num_fut=10
#aa = anomaly_uni_LSTM(list_var,num_fut)
#print aa


    
