import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
import math
import helpers as h


def moving_test_window_preds(n_future_preds):

    ''' n_future_preds - Represents the number of future predictions we want to make
                         This coincides with the number of windows that we will move forward
                         on the test data
    '''
    preds_moving = []                                    # Use this to store the prediction made on each test window
    moving_test_window = [test_X[0,:].tolist()]          # Creating the first test window
    moving_test_window = np.array(moving_test_window)    # Making it an numpy array
    
    for i in range(n_future_preds):
        preds_one_step = model.predict(moving_test_window) # Note that this is already a scaled prediction so no need to rescale this
        preds_moving.append(preds_one_step[0,0]) # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1,1,1) # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:,1:,:], preds_one_step), axis=1) # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end
        
    preds_moving = scaler.inverse_transform(preds_moving)
    
    return preds_moving




def anomaly_uni_LSTM(lista_datos,desv_mse=0):
    temp= pd.DataFrame(lista_datos,columns=['values'])
    data_raw = temp.values.astype("float32")

    scaler = MinMaxScaler(feature_range = (0, 1))
    dataset = scaler.fit_transform(data_raw)


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

    pred_scaled =model.predict(forecast_X)
    pred = scaler.inverse_transform(pred_scaled)
    
    print ("el tamano de la preddicion")
    print (len(pred))

    print(pred)
    print ('prediccion')

    engine_output={}


    engine_output['rmse'] = str(math.sqrt(mse))
    engine_output['mse'] = int(mse)
    engine_output['mae'] = int(mae)
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='LSTM'
    df_future= pd.DataFrame(pred[len(pred) - 5:],columns=['value'])
    df_future['value']=df_future.value.astype("float64")
    df_future['step']= np.arange( len(lista_datos),len(lista_datos)+5,1)
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



def anomaly_LSTM(list_var,desv_mse=0):
    
    df_var = pd.DataFrame()
    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]

    values = df_var.values
    scaled = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)


    # split into train and test sets
    values = reframed.values

    TRAIN_SIZE = 0.70
    train_size = int(len(values) * TRAIN_SIZE)
    test_size = len(values) - train_size
    train, test = values[0:train_size, :], values[train_size-2:len(values), :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), shuffle=False)



    # make a prediction
    yhat = model.predict(test_X)

    #pyplot.plot(test_y, label='real')
    #pyplot.plot(yhat, label='pred')
    #pyplot.legend()
    #pyplot.show()


    test_y = test_y.reshape((len(test_y), 1))

    #inv_y = concatenate((test_y, yhat), axis=1)
    #inv_y = scaler.inverse_transform(inv_y)

    test_y = ((test_y[:,0] * (test_y[:,0].max()-(test_y[:,0].min()))) + test_y[:,0].min())*100
    yhat = ((yhat[:,0] * (yhat[:,0].max()-yhat[:,0].min())) + yhat[:,0].min())*100

    #pyplot.plot(test_y, label='aa')
    #pyplot.plot(yhat, label='bb')
    #pyplot.legend()
    #pyplot.show()

    #test_y = inv_y[:,0]
    #yhat = inv_y[:,1]

    #print test_y[:,0]
    #desn_1 = ((test_y[:,0] * (max(test_y[:,0])-min(test_y[:,0]))) + min(test_y[:,0]))*100
    #desn_2 = ((yhat[:,0] * (max(yhat[:,0])-min(yhat[:,0]))) + min(yhat[:,0]))*100


    mse = (mean_squared_error(test_y, yhat))
    rmse = np.sqrt(mse)
    df_aler = pd.DataFrame()
    print 'mse', mse
    print 'rmse', rmse

    print test_y
    print yhat


    lista_puntos = np.arange(train_size -2, train_size + test_size,1)
    testing_data = pd.DataFrame(yhat,index =lista_puntos,columns=['expected value'])


    df_aler['real_value'] = test_y
    df_aler['expected_value'] = yhat

    df_aler['mse'] = mse
    df_aler['puntos'] = df_aler.index
    df_aler['puntos'] = df_aler['puntos'] + train_size
    df_aler.set_index('puntos',inplace=True)


    df_aler['rmse'] = rmse
    mae = mean_absolute_error(yhat, test_y)
    df_aler['mae'] = mean_absolute_error(yhat, test_y)


    df_aler['anomaly_score'] = abs(df_aler['expected_value']-df_aler['real_value'])/df_aler['mae']
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

    print df_aler_ult

    #forecast

    values = df_var.values
    scaled = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)


    values = reframed.values

    num_fut=5

    test1_X, test1_y = values[:, :-1], values[:, -1]

    test1_X = test1_X.reshape((test1_X.shape[0], 1, test1_X.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(test1_X.shape[1], test1_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    history = model.fit(test1_X, test1_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

    len_fore = len(test1_X) - num_fut
    fore = test1_X[len_fore:]
    yhat = model.predict(fore)

    print yhat
    test1_y = ((test1_y * (test1_y.max()-(test1_y.min()))) + test1_y.min())*100
    yhat = ((yhat[:,0] * (yhat[:,0].max()-yhat[:,0].min())) + yhat[:,0].min())*100

    #pyplot.plot(yhat, label='fut')
    #pyplot.legend()
    #pyplot.show()


    lista_result = np.arange(len(test1_X), (len(test1_X)+num_fut),1)
    df_result_forecast = pd.DataFrame({'puntos':lista_result, 'valores':yhat})
    df_result_forecast.set_index('puntos',inplace=True)
    df_result_forecast['valores']=df_result_forecast['valores'].astype(str)
    df_result_forecast['step'] = df_result_forecast.index


    engine_output={}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='LSTM'
    engine_output['future']= df_result_forecast.fillna(0).to_dict(orient='record')

    testing_data['step']=testing_data.index
    engine_output['debug'] = testing_data.fillna(0).to_dict(orient='record')
    print engine_output

    return (engine_output)



def forecast_LSTM(list_var,num_fut):
    
    df_var = pd.DataFrame()
    
    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]
    
    values = df_var.values

    test_X, test_y = values[:, :-1], values[:, -1]

    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(test_X.shape[1], test_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    
    history = model.fit(test_X, test_y, epochs=50, batch_size=72, verbose=0, shuffle=False)
    
    len_fore = len(test_X) - num_fut
    fore = test_X[len_fore:]
    yhat = model.predict(fore)
    
    lista_result = np.arange(len(test_X), (len(test_X)+num_fut),1)
    df_result = pd.DataFrame({'puntos':lista_result, 'valores':yhat[:,0]})
    df_result.set_index('puntos',inplace=True)
    return (df_result)
