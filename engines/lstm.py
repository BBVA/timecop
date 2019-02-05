
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras import backend as K
import math
#import helpers as h
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense

import gc

import math
from matplotlib import pyplot
from numpy.random import seed
seed(69)
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import pickle

#import multiprocessing
from . BBDD import new_model, get_best_model







def add_hlayer(model, num_nodes, return_sequences=False):
    model.add(LSTM(num_nodes, return_sequences=return_sequences))

def define_model(n_nodes, n_hlayers, dropout, input_data, output_shape):
    model = Sequential()
    if n_hlayers == 1:
        model.add(LSTM(output_dim =int(n_nodes), activation='relu', input_shape =(input_data.shape[1], input_data.shape[2]),
                   return_sequences=False))
    else:
        #model.add(LSTM(output_dim =int(n_nodes), activation='relu', input_shape =(input_data.shape[1], input_data.shape[2]),return_sequences=True))
        model.add(LSTM(activation='relu', input_shape =(input_data.shape[1], input_data.shape[2]),return_sequences=True,units =int(n_nodes) ))
    model.add(Dropout(dropout))
    #print(n_hlayers)

    for i in range(n_hlayers-1):
        #print(i)
        if i == n_hlayers-2:
            #add_hlayer(model, n_nodes, return_sequences=False)
            model.add(LSTM(n_nodes, return_sequences=False))
            model.add(Dropout(dropout))
            model.add(BatchNormalization())
        else:
            #add_hlayer(model, n_nodes, return_sequences=True)
            model.add(LSTM(n_nodes, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(BatchNormalization())

    model.add(Dense(int(n_nodes/2), activation='relu'))
    model.add(Dropout(dropout))

    #model.add(Dense(output_dim=int(output_shape)))
    model.add(Dense(units=int(output_shape)))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def hyperparameter_opt(list_hlayers, list_n_nodes, n_dropout, input_data, output_shape):
    models_dict = {}
    for hlayer in list_hlayers:
        for nodes in list_n_nodes:
            for drop in n_dropout:
                model = define_model(nodes, hlayer, drop, input_data, output_shape)
                name = 'model_nlayers_{}_nnodes_{}_dropout_{}'.format(hlayer, nodes, drop)
                models_dict[name] = model
                print(name)

    return models_dict

def anomaly_uni_LSTM(lista_datos,num_forecast=10,desv_mse=2,train='True',name='test'):

    temp= pd.DataFrame(lista_datos,columns=['values'])

    scaler_x = MinMaxScaler(feature_range =(-1, 1))
    x = np.array(temp)
    #pyplot.plot(x, label='pred')
    #pyplot.legend()
    #pyplot.show()

    x = scaler_x.fit_transform(temp)
    x = x[:,0]
    print ('x',x)
    TRAIN_SIZE = 0.7
    train_size = int(len(x) * TRAIN_SIZE)
    test_size = len(x) - train_size

    x_train, x_test = x[0:train_size], x[train_size:len(x)]
    print ('x_train',x_train)
    print ('x_test',x_test)


    window_size = 1

    num_fore = num_forecast + 1

    win_train_x, win_train_y = [], []
    for i in range(len(x_train) - window_size - 1):
        if len(x_train)<(i+num_fore):
            break
        a = x_train[i:(i + window_size)]
        win_train_x.append(a)
        win_train_y.append(x_train[i + window_size: i+num_fore])


    win_train_x = np.array(win_train_x)
    print ('win_train_x',win_train_x)
    print ('shape win_train_x',win_train_x.shape)
    win_train_y = np.array(win_train_y)
    print ('win_train_y',win_train_y)
    print ('shape win_train_y',win_train_y.shape)


    win_train_x = win_train_x.reshape((win_train_x.shape[0], 1, win_train_x.shape[1]))
    print('reshape win_train_x',win_train_x.shape)
    new_test_x = x_test.reshape((x_test.shape[0], 1, 1))
    print ('new_test_x',new_test_x)


    actual_model=''


    ############### hyperparameter finding

    if (train):

        ##################neural network######################

        models_dict = {}
        n_hlayers = [1, 2]
        n_nodes = [100, 300, 500]
        n_dropout = [0, 0.1, 0.15, 0.20]

        #pruebas
        #n_hlayers = [1]
        #n_nodes = [500]
        #n_dropout = [0.15]

        # models_dict = hyperparameter_opt(n_hlayers, n_nodes, n_dropout, win_train_x, num_forecast)
        #
        # for model in models_dict:
        #     print(model)
        #     print(models_dict[model].summary())
        #
        # print ('Numero de modelos',len(models_dict))




##########################################################################################
#############################################################################################3
        best_mae = 999999999
        best_model=''
        for hlayer in n_hlayers:
            for nodes in n_nodes:
                for drop in n_dropout:
                    K.clear_session()
                    gc.collect()
                    model = define_model(nodes, hlayer, drop, win_train_x, num_forecast)
                    model_name = 'model_nlayers_{}_nnodes_{}_dropout_{}'.format(hlayer, nodes, drop)
                    model.fit(win_train_x, win_train_y, epochs=65, verbose=0, shuffle=False)

                    #models_dict[name] = model
                    print(model_name)
                    yhat = model.predict(new_test_x)
                    yhat_test = yhat[:,0]

                    temp_res= pd.DataFrame(yhat_test,columns=['values'])
                    temp_res = np.array(temp_res)
                    y_yhat_inv = scaler_x.inverse_transform(temp_res)
                    y_yhat_inv= y_yhat_inv[:,0]

                    temp_x_test= pd.DataFrame(x_test,columns=['values'])
                    temp_x_test = np.array(temp_x_test)
                    x_test_inv = scaler_x.inverse_transform(temp_x_test)

                    mse = (mean_squared_error(x_test_inv, y_yhat_inv))
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(x_test_inv, y_yhat_inv)
                    print ('mse', mse)
                    print ('rmse', rmse)
                    print ('mae', mae)
                    if mae < best_mae:
                            best_model=model







        #####getting best model
    #     #dict_eval_models = {}
    #     dict_mse_models = {}
    #     for model in models_dict:
    # #        print 'fit model {}'.format(model)
    #         try:
    #             seed(69)
    #             #name_model = models_dict[model].fit(win_train_x, win_train_y, epochs=25, verbose=0, shuffle=False)
    #             models_dict[model].fit(win_train_x, win_train_y, epochs=25, verbose=0, shuffle=False)
    #             #dict_eval_models[model] = name_model
    #         except:
    #             dict_eval_models[model] = 'Error'
    #
    #
    #         print(model)
    #         yhat = models_dict[model].predict(new_test_x)
    #         yhat_test = yhat[:,0]
    #
    #         temp_res= pd.DataFrame(yhat_test,columns=['values'])
    #         temp_res = np.array(temp_res)
    #         y_yhat_inv = scaler_x.inverse_transform(temp_res)
    #         y_yhat_inv= y_yhat_inv[:,0]
    #
    #         temp_x_test= pd.DataFrame(x_test,columns=['values'])
    #         temp_x_test = np.array(temp_x_test)
    #         x_test_inv = scaler_x.inverse_transform(temp_x_test)
    #
    #         mse = (mean_squared_error(x_test_inv, y_yhat_inv))
    #         rmse = np.sqrt(mse)
    #         mae = mean_absolute_error(x_test_inv, y_yhat_inv)
    #         print ('mse', mse)
    #         print ('rmse', rmse)
    #         print ('mae', mae)
    #         dict_mse_models[model] = mae
    #         # if mae != min(dict_mse_models, key = dict_mse_models.get):
    #         #     del dict_mse_models[model]
    #         #     del models_dict[model]
    #
    #     best_model = min(dict_mse_models, key = dict_mse_models.get)


        #print('best_model',best_model)
        #K.clear_session()
        # for model in models_dict:
        #     if model != best_model:
        #         del models_dict[model]
        #         print ("Model "+ model +" erased")
        gc.collect()

        best_model.save('./models_temp/lstm.model'+name)
        print ("insertando modelo LSTM")
        with open('./models_temp/lstm.model'+name,'rb') as f:
            mymodel = f.read()

            new_model(name, 'LSTM', bytearray(mymodel),'',best_mae)
            f.close()
        actual_model= best_model

    else:

        print ("Adquiring best LSTM model")
        (model_name,mymodel,params)=get_best_model(name)
        #print("el modelo es")
        #print(model_name)
        #print (mymodel)
        with open('./models_temp/lstm.model'+name, "wb") as newFile:
            newFile.write(mymodel)
            newFile.close()


        actual_model= load_model('./models_temp/lstm.model'+name)

    yhat = actual_model.predict(new_test_x)
    print ('yhat',yhat)

    ###################################save the best model

    #model_filename = "./models_temp/lstm_model"
    #models_dict[best_model].save(model_filename)

    yhat_test = yhat[:,0]
    print ('yhat_test',yhat_test)

    temp_res= pd.DataFrame(yhat_test,columns=['values'])
    temp_res = np.array(temp_res)

    y_yhat_inv = scaler_x.inverse_transform(temp_res)
    print ('y_yhat_inv',y_yhat_inv)

    y_yhat_inv= y_yhat_inv[:,0]

    temp_x_test= pd.DataFrame(x_test,columns=['values'])
    temp_x_test = np.array(temp_x_test)
    x_test_inv = scaler_x.inverse_transform(temp_x_test)
    x_test_inv= x_test_inv[:,0]
    print ('x_test_inv',x_test_inv)

    #pyplot.plot(x_test_inv, label='real')
    #pyplot.plot(y_yhat_inv, label='pred')
    #pyplot.legend()
    #pyplot.show()

    mse = (mean_squared_error(x_test_inv, y_yhat_inv))
    rmse = np.sqrt(mse)
    print ('mse', mse)
    print ('rmse', rmse)

    df_aler = pd.DataFrame()
    lista_puntos = np.arange(train_size, train_size + test_size,1)
    testing_data = pd.DataFrame(y_yhat_inv,index =lista_puntos,columns=['expected value'])

    #print ('x_test_inv',x_test_inv)
    #print ('y_yhat_inv',y_yhat_inv)

    df_aler['real_value'] = x_test_inv
    df_aler['expected_value'] = y_yhat_inv

    df_aler['mse'] = mse
    df_aler['puntos'] = df_aler.index
    df_aler['puntos'] = df_aler['puntos'] + train_size
    df_aler.set_index('puntos',inplace=True)


    df_aler['rmse'] = rmse
    mae = mean_absolute_error(y_yhat_inv, x_test_inv)
    df_aler['mae'] = mae


    df_aler['anomaly_score'] = abs(df_aler['expected_value']-df_aler['real_value'])/df_aler['mae']
    #print (df_aler)

    df_aler_ult = df_aler[:5]
    df_aler = df_aler[(df_aler['anomaly_score']> desv_mse)]
    max_anom = df_aler['anomaly_score'].max()
    min_anom = df_aler['anomaly_score'].min()

    df_aler['anomaly_score'] = ( df_aler['anomaly_score'] - min_anom ) /(max_anom - min_anom)

    print ('Anomaly')
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

    #print (df_aler_ult)



    ################## forecast
    win_todo_x, win_todo_y = [], []
    for i in range(len(x) - window_size - 1):
        if len(x)<(i+num_fore):
            break
        a = x[i:(i + window_size)]
        win_todo_x.append(a)
        win_todo_y.append(x[i + window_size: i+num_fore])

    win_todo_x = np.array(win_todo_x)
    #print ('win_todo_x',win_todo_x)
    #print ('shape win_todo_x',win_todo_x.shape)

    win_todo_y = np.array(win_todo_y)
    #print ('win_todo_y',win_todo_y)
    #print ('shape win_todo_y',win_todo_y.shape)

    win_todo_x = win_todo_x.reshape((win_todo_x.shape[0], 1, win_todo_x.shape[1]))
    #print('reshape win_todo_x',win_todo_x.shape)


    name_model = actual_model.fit(win_todo_x, win_todo_y, epochs=25, verbose=0, shuffle=False)



    falta_win_todo_x = x[-num_forecast:]
    #print ('falta_win_todo_x',falta_win_todo_x)
    #print ('shape falta_win_todo_x',falta_win_todo_x.shape)

    falta_win_todo_x = falta_win_todo_x.reshape(falta_win_todo_x.shape[0],1,1)
    #print ('x',x)
    #print ('falta_win_todo_x',falta_win_todo_x)
    yhat_todo = actual_model.predict(falta_win_todo_x)
    #print ('yhat_todo',yhat_todo)
    #print ('yhat_todo',yhat_todo[-1,:])

    temp_res= pd.DataFrame(yhat_todo[-1],columns=['values'])
    temp_res = np.array(temp_res)
    y_fore_inv = scaler_x.inverse_transform(temp_res)

    y_fore_inv= y_fore_inv[:,0]


    #pyplot.plot(y_fore_inv, label='pred')
    #pyplot.legend()
    #pyplot.show()


    engine_output={}

    engine_output['rmse'] = int(rmse)
    engine_output['mse'] = int(mse)
    engine_output['mae'] = int(mae)
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='LSTM'

    df_future= pd.DataFrame(y_fore_inv,columns=['value'])

    df_future['value']=df_future.value.astype("float64")
    df_future['step']= np.arange(len(x),len(x)+len(y_fore_inv),1)
    engine_output['future'] = df_future.fillna(0).to_dict(orient='record')

    testing_data['step']=testing_data.index

    engine_output['debug'] = testing_data.fillna(0).to_dict(orient='record')
    K.clear_session()
    for model in models_dict:
        del models_dict[model]
    print ("Models erased")
    gc.collect()
    return engine_output



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






def anomaly_LSTM(list_var,num_fut=10,desv_mae=2):

    df_var = pd.DataFrame()
    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]
    #print df_var

    temp_var_ult = pd.DataFrame(df_var[df_var.columns[-1]])
    scaler_y = MinMaxScaler(feature_range =(-1, 1))
    y = scaler_y.fit_transform(temp_var_ult)
    #print ('y', y)

    scaler_x = MinMaxScaler(feature_range =(-1, 1))
    #x = np.array(df_var)
    #print x
    x = scaler_x.fit_transform(df_var)
    #x = x[:,0]
    #print ('x',x)

    TRAIN_SIZE = 0.7
    train_size = int(len(x) * TRAIN_SIZE)
    test_size = len(x) - train_size

    x_train, x_test = x[0:train_size], x[train_size:len(x)]
    #print ('x_train',x_train)
    #print ('x_test',x_test)
    #print ('shape x_test',x_test.shape)

    window_size = 1
    num_fore = num_forecast + 1

    win_train_x, win_train_y = [], []
    for i in range(len(x_train) - window_size - 1):
        if len(x_train)<(i+num_fore):
            break
        a = x_train[i:(i + window_size)]
        win_train_x.append(a)
        win_train_y.append(x_train[i + window_size: i+num_fore])

    win_train_x = np.array(win_train_x)
    print ('win_train_x',win_train_x)
    print ('shape win_train_x',win_train_x.shape)
    win_train_y = np.array(win_train_y)
    print ('win_train_y',win_train_y)
    print ('shape win_train_y',win_train_y.shape)
    win_train_y_var_pred = win_train_y[:,:,-1]
    print ('win_train_y_var_pred',win_train_y_var_pred)
    print ('shape win_train_y_var_pred',win_train_y_var_pred.shape)

    new_test_x = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    print ('new_test_x',new_test_x)
    print ('shape new_test_x',new_test_x.shape)



    ##################neural network######################

    models_dict = {}
    n_hlayers = [1, 2, 3]
    n_nodes = [100, 300, 500,700]
    n_dropout = [0, 0.1, 0.15, 0.20]

    #pruebas
    #n_hlayers = [1]
    #n_nodes = [500]
    #n_dropout = [0]

    models_dict = hyperparameter_opt(n_hlayers, n_nodes, n_dropout, win_train_x, num_forecast)

    for model in models_dict:
        print(model)
        print(models_dict[model].summary())

    print ('Numero de modelos',len(models_dict))

    #####getting best model
    dict_eval_models = {}
    for model in models_dict:
        #print 'fit model {}'.format(model)
        try:
            seed(69)
            name_model = models_dict[model].fit(win_train_x, win_train_y_var_pred, epochs=25, verbose=0, shuffle=False)
            dict_eval_models[model] = name_model
        except:
            dict_eval_models[model] = 'Error'


    dict_mse_models = {}
    for model in models_dict:
        print(model)
        yhat = models_dict[model].predict(new_test_x)
        yhat_test = yhat[:,0]

        temp_res= pd.DataFrame(yhat_test,columns=['values'])
        temp_res = np.array(temp_res)
        y_yhat_inv = scaler_y.inverse_transform(temp_res)
        y_yhat_inv= y_yhat_inv[:,0]

        x_test_var_pred = x_test[:,-1]
        #print ('x_test_var_pred', x_test_var_pred)
        temp_x_test= pd.DataFrame(x_test_var_pred,columns=['values'])
        temp_x_test = np.array(temp_x_test)
        x_test_inv = scaler_y.inverse_transform(temp_x_test)
        x_test_inv= x_test_inv[:,0]

        #pyplot.plot(x_test_inv, label='real')
        #pyplot.plot(y_yhat_inv, label='pred')
        #pyplot.legend()
        #pyplot.show()

        mse = (mean_squared_error(x_test_inv, y_yhat_inv))
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_yhat_inv, x_test_inv)
        print ('mse', mse)
        print ('rmse', rmse)
        print ('mae', mae)
        dict_mse_models[model] = mae

    best_model = min(dict_mse_models, key = dict_mse_models.get)


    print('best_model',best_model)
    yhat = models_dict[best_model].predict(new_test_x)
    yhat_test = yhat[:,0]

    temp_res= pd.DataFrame(yhat_test,columns=['values'])
    temp_res = np.array(temp_res)
    y_yhat_inv = scaler_y.inverse_transform(temp_res)
    y_yhat_inv= y_yhat_inv[:,0]

    x_test_var_pred = x_test[:,-1]
    #print ('x_test_var_pred', x_test_var_pred)
    temp_x_test= pd.DataFrame(x_test_var_pred,columns=['values'])
    temp_x_test = np.array(temp_x_test)
    x_test_inv = scaler_y.inverse_transform(temp_x_test)
    x_test_inv= x_test_inv[:,0]

    #pyplot.plot(x_test_inv, label='real')
    #pyplot.plot(y_yhat_inv, label='pred')
    #pyplot.legend()
    #pyplot.show()

    mse = (mean_squared_error(x_test_inv, y_yhat_inv))
    rmse = np.sqrt(mse)
    print ('mse', mse)
    print ('rmse', rmse)



    df_aler = pd.DataFrame()
    lista_puntos = np.arange(train_size, train_size + test_size,1)
    testing_data = pd.DataFrame(y_yhat_inv,index =lista_puntos,columns=['expected value'])

    #print ('x_test_inv',x_test_inv)
    #print ('y_yhat_inv',y_yhat_inv)

    df_aler['real_value'] = x_test_inv
    df_aler['expected_value'] = y_yhat_inv

    df_aler['mse'] = mse
    df_aler['puntos'] = df_aler.index
    df_aler['puntos'] = df_aler['puntos'] + train_size
    df_aler.set_index('puntos',inplace=True)


    df_aler['rmse'] = rmse
    mae = mean_absolute_error(y_yhat_inv, x_test_inv)
    df_aler['mae'] = mae


    df_aler['anomaly_score'] = abs(df_aler['expected_value']-df_aler['real_value'])/df_aler['mae']
    print (df_aler)

    df_aler_ult = df_aler[:5]
    df_aler = df_aler[(df_aler['anomaly_score']> desv_mse)]
    max_anom = df_aler['anomaly_score'].max()
    min_anom = df_aler['anomaly_score'].min()

    df_aler['anomaly_score'] = ( df_aler['anomaly_score'] - min_anom ) /(max_anom - min_anom)

    #print ('Anomaly')
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

    #print (df_aler_ult)

    ###forecast
    win_todo_x, win_todo_y = [], []
    for i in range(len(x) - window_size - 1):
        if len(x)<(i+num_fore):
            break
        a = x[i:(i + window_size)]
        win_todo_x.append(a)
        win_todo_y.append(x[i + window_size: i+num_fore])

    win_todo_x = np.array(win_todo_x)
    #print ('win_todo_x',win_todo_x)
    #print ('shape win_todo_x',win_todo_x.shape)

    win_todo_y = np.array(win_todo_y)
    #print ('win_todo_y',win_todo_y)
    #print ('shape win_todo_y',win_todo_y.shape)

    win_todo_y_var_pred = win_todo_y[:,:,-1]
    #print ('win_todo_y_var_pred',win_todo_y_var_pred)
    #print ('shape win_todo_y_var_pred',win_todo_y_var_pred.shape)

    name_model = models_dict[best_model].fit(win_todo_x, win_todo_y_var_pred, epochs=25, verbose=0, shuffle=False)

    falta_win_todo_x = x[-num_forecast:,:]
    #print ('falta_win_todo_x',falta_win_todo_x)
    #print ('shape falta_win_todo_x',falta_win_todo_x.shape)

    falta_win_todo_x = falta_win_todo_x.reshape(falta_win_todo_x.shape[0],1,falta_win_todo_x.shape[1])
    #print ()'x',x)
    #print ('falta_win_todo_x',falta_win_todo_x)

    yhat_todo = models_dict[best_model].predict(falta_win_todo_x)
    #print ('yhat_todo',yhat_todo)
    #print ('yhat_todo',yhat_todo[-1,:])

    temp_res= pd.DataFrame(yhat_todo[-1],columns=['values'])
    temp_res = np.array(temp_res)
    y_fore_inv = scaler_y.inverse_transform(temp_res)
    y_fore_inv= y_fore_inv[:,0]

    #pyplot.plot(y_fore_inv, label='pred')
    #pyplot.legend()
    #pyplot.show()

    engine_output={}

    engine_output['rmse'] = int(rmse)
    engine_output['mse'] = int(mse)
    engine_output['mae'] = int(mae)
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='LSTM'

    df_future= pd.DataFrame(y_fore_inv,columns=['value'])

    df_future['value']=df_future.value.astype("float64")
    df_future['step']= np.arange(len(x),len(x)+len(y_fore_inv),1)
    engine_output['future'] = df_future.to_dict(orient='record')

    testing_data['step']=testing_data.index

    engine_output['debug'] = testing_data.to_dict(orient='record')

    return engine_output





#list_var=[5938,6925,4353,6855,6216,7061,4417,7505,6778,8169,5290,7710,6213,7952,5476,7990,7747,8908,5369,7794,7543,8533,5564,6997,6678,7730,4790,6027,5877,7413,4459,5966,5305,6238,3689,5146,4799,6044,3695,4795,4147,5254,2679,3428,3635,5391,3142,3986,3807,4732,2513,3834,3133,4405]

#num_fut=10
#aa = anomaly_uni_LSTM(list_var,num_fut)
#print aa
