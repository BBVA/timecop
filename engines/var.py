import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pyflux as pf
#from helpers import helpers as h
from . BBDD import new_model, get_best_model
from struct import *




def univariate_anomaly_VAR(lista_datos,num_fut,name):
    lista_puntos = np.arange(0, len(lista_datos),1)


    df = pd.DataFrame()
    df['valores'] = lista_datos

    df['valores'] = df.valores.astype(np.float)

    tam_train = int(len(df)*0.7)
    #print tam_train
    df_train = df[:tam_train]
    print('Tamanio train: {}'.format(df_train.shape))
    df_test = df[tam_train:]
    print('Tamanio test: {}'.format(df_test.shape))

    print (type(df_test))
    mae_period = 99999999
    best_lag=0
    lags = int(round(len(df_train)/2))
    print ("empezamos el bucle")
    for lag in range(lags):
        model = pf.VAR(df_train,lags=lag)
        x = model.fit()


        print ("fit ready")
        future_forecast_pred = model.predict(len(df_test))
        future_forecast_pred = future_forecast_pred[['valores']]

        list_test = df_test['valores'].values
        list_future_forecast_pred = future_forecast_pred['valores'].values

        #pyplot.plot(list_test, label='real')
        #pyplot.plot(list_future_forecast_pred, label='pred')
        #pyplot.legend()
        #pyplot.show()

        mae_temp = mean_absolute_error(list_test, list_future_forecast_pred)
        print('El error medio del modelo_test es: {}'.format(mae_temp))

        if mae_temp < mae_period:
            best_lag=lag
            mae_period=mae_temp
        else:
            print ("mae:" + str(mae_period))

    print ("######best mae is " + str(mae_period) + " with the lag " + str(best_lag))

    model = pf.VAR(df_train,lags=best_lag)
    x = model.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))
    #model.plot_predict_is(h=8, figsize=((8,5)))
    #model.plot_predict(past_values=20, h=6, figsize=(8,5))

    future_forecast_pred = model.predict(len(df_test))
    future_forecast_pred = future_forecast_pred[['valores']]

    list_test = df_test['valores'].values
    list_future_forecast_pred = future_forecast_pred['valores'].values

    #mse_test = (list_future_forecast_pred - list_test)
    #mse_abs_test = abs(mse_test)

    #pyplot.plot(list_test, label='real')
    #pyplot.plot(list_future_forecast_pred, label='pred')
    #pyplot.legend()
    #pyplot.show()

    mse = mean_squared_error(list_test, list_future_forecast_pred)
    print('El error medio del modelo_test es: {}'.format(mse))


    rmse = np.sqrt(mse)
    print('El root error medio del modelo_test es: {}'.format(rmse))
    mae = mean_absolute_error(list_test, list_future_forecast_pred)


    print ("Saving params")
    filename = './models_temp/learned_model_var'+name
    with open(filename,'w') as f:
        f.write(str(best_lag))
        f.close()

    print ("insertando modelo VAR")
    new_model(name, 'VAR', pack('N', 365),str(best_lag))



    df_aler = pd.DataFrame()
    df_aler['real_value'] = list_test
    df_aler['expected value'] = list_future_forecast_pred
    df_aler['mse'] = mse
    df_aler['puntos'] = future_forecast_pred.index
    df_aler.set_index('puntos',inplace=True)
    df_aler['mae'] = mae

    df_aler['anomaly_score'] = abs(df_aler['expected value']-df_aler['real_value'])/df_aler['mae']

    df_aler = df_aler[(df_aler['anomaly_score']> 2)]

    max = df_aler['anomaly_score'].max()
    min = df_aler['anomaly_score'].min()
    df_aler['anomaly_score']= ( df_aler['anomaly_score'] - min ) /(max - min)

    df_aler_ult = df_aler[:5]
    df_aler_ult = df_aler_ult[(df_aler_ult.index==df_aler.index.max())|(df_aler_ult.index==((df_aler.index.max())-1))
                             |(df_aler_ult.index==((df_aler.index.max())-2))|(df_aler_ult.index==((df_aler.index.max())-3))
                             |(df_aler_ult.index==((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = 'FALSE'
    else:
        exists_anom_last_5 = 'TRUE'

    max = df_aler_ult['anomaly_score'].max()
    min = df_aler_ult['anomaly_score'].min()
    print (df_aler_ult)
    df_aler_ult['anomaly_score'] = ( df_aler_ult['anomaly_score'] - min ) /(max - min)

    #####forecast#####

    model_for = pf.VAR(df,lags=best_lag)
    x_for = model_for.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))

    future_forecast_pred_for = model_for.predict(num_fut)

    #pyplot.plot(future_forecast_pred_for, label='forecast')
    #pyplot.legend()
    #pyplot.show()

    df_result_forecast = future_forecast_pred_for.reset_index()
    df_result_forecast = df_result_forecast.rename(columns = {'index':'step'})

    print (df.head(5))
    print (df.tail(5))

    engine_output={}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.to_dict(orient='record')
    engine_output['engine']='VAR'
    engine_output['future']= df_result_forecast.fillna(0).to_dict(orient='record')
    test_values = pd.DataFrame(future_forecast_pred.values,index = df_test.index,columns=['expected value'])
    test_values['step'] = test_values.index
    engine_output['debug'] = test_values.fillna(0).to_dict(orient='record')


    return (engine_output)













def univariate_forecast_VAR(lista_datos,num_fut,name):
    lista_puntos = np.arange(0, len(lista_datos),1)


    df = pd.DataFrame()
    df['valores'] = lista_datos

    df['valores'] = df.valores.astype(np.float)

    tam_train = int(len(df)*0.7)
    #print tam_train
    df_train = df[:tam_train]
    print('Tamanio train: {}'.format(df_train.shape))
    df_test = df[tam_train:]
    print('Tamanio test: {}'.format(df_test.shape))

    print (type(df_test))
    mae_period = 99999999
    best_lag=0
    lags = int(round(len(df_train)/2))

    filename = './models_temp/learned_model_var'+name

    with open(filename,'r') as f:

        best_lag = int(f.read())
        f.close()


    (model_name,model,params)=get_best_model(name)
    best_lag = int(params)


    model = pf.VAR(df_train,lags=best_lag)
    x = model.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))
    #model.plot_predict_is(h=8, figsize=((8,5)))
    #model.plot_predict(past_values=20, h=6, figsize=(8,5))

    future_forecast_pred = model.predict(len(df_test))
    future_forecast_pred = future_forecast_pred[['valores']]

    list_test = df_test['valores'].values
    list_future_forecast_pred = future_forecast_pred['valores'].values

    #mse_test = (list_future_forecast_pred - list_test)
    #mse_abs_test = abs(mse_test)

    #pyplot.plot(list_test, label='real')
    #pyplot.plot(list_future_forecast_pred, label='pred')
    #pyplot.legend()
    #pyplot.show()

    mse = mean_squared_error(list_test, list_future_forecast_pred)
    print('El error medio del modelo_test es: {}'.format(mse))


    rmse = np.sqrt(mse)
    print('El root error medio del modelo_test es: {}'.format(rmse))
    mae = mean_absolute_error(list_test, list_future_forecast_pred)




    df_aler = pd.DataFrame()
    df_aler['real_value'] = list_test
    df_aler['expected value'] = list_future_forecast_pred
    df_aler['mse'] = mse
    df_aler['puntos'] = future_forecast_pred.index
    df_aler.set_index('puntos',inplace=True)
    df_aler['mae'] = mae

    df_aler['anomaly_score'] = abs(df_aler['expected value']-df_aler['real_value'])/df_aler['mae']

    df_aler = df_aler[(df_aler['anomaly_score']> 2)]

    max = df_aler['anomaly_score'].max()
    min = df_aler['anomaly_score'].min()
    df_aler['anomaly_score']= ( df_aler['anomaly_score'] - min ) /(max - min)

    df_aler_ult = df_aler[:5]
    df_aler_ult = df_aler_ult[(df_aler_ult.index==df_aler.index.max())|(df_aler_ult.index==((df_aler.index.max())-1))
                             |(df_aler_ult.index==((df_aler.index.max())-2))|(df_aler_ult.index==((df_aler.index.max())-3))
                             |(df_aler_ult.index==((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = 'FALSE'
    else:
        exists_anom_last_5 = 'TRUE'

    max = df_aler_ult['anomaly_score'].max()
    min = df_aler_ult['anomaly_score'].min()
    print (df_aler_ult)
    df_aler_ult['anomaly_score'] = ( df_aler_ult['anomaly_score'] - min ) /(max - min)

    #####forecast#####

    model_for = pf.VAR(df,lags=best_lag)
    x_for = model_for.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))

    future_forecast_pred_for = model_for.predict(num_fut)

    #pyplot.plot(future_forecast_pred_for, label='forecast')
    #pyplot.legend()
    #pyplot.show()

    df_result_forecast = future_forecast_pred_for.reset_index()
    df_result_forecast = df_result_forecast.rename(columns = {'index':'step'})

    print (df.head(5))
    print (df.tail(5))

    engine_output={}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.to_dict(orient='record')
    engine_output['engine']='VAR'
    engine_output['future']= df_result_forecast.fillna(0).to_dict(orient='record')
    test_values = pd.DataFrame(future_forecast_pred.values,index = df_test.index,columns=['expected value'])
    test_values['step'] = test_values.index
    engine_output['debug'] = test_values.fillna(0).to_dict(orient='record')


    return (engine_output)



def anomaly_VAR(list_var,num_fut):
    df_var = pd.DataFrame()

    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]
        df_var['var_{}'.format(i)] = list_var[i]


    df_var.rename(columns = {df_var.columns[-1]:'expected value'},inplace=True)
    tam_train = int(len(df_var)*0.7)
    #print tam_train
    df_train = df_var[:tam_train]
    print('Tamanio train: {}'.format(df_train.shape))
    df_test = df_var[tam_train:]
    print('Tamanio test: {}'.format(df_test.shape))


    mae_period = 99999999
    best_lag=0
    lags = int(round(len(df_train)/2))
    if (lags > 100):
        lags=100
    for lag in range(lags):
        print ("entra en el bucle con dato " + str(lag))
        model = pf.VAR(df_train,lags=lag)
        x = model.fit()


        future_forecast_pred = model.predict(len(df_test))
        future_forecast_pred = future_forecast_pred[['expected value']]

        list_test = df_test['expected value'].values
        list_future_forecast_pred = future_forecast_pred['expected value'].values

        #pyplot.plot(list_test, label='real')
        #pyplot.plot(list_future_forecast_pred, label='pred')
        #pyplot.legend()
        #pyplot.show()

        mae_temp = mean_absolute_error(list_test, list_future_forecast_pred)
        print('El error medio del modelo_test es: {}'.format(mae_temp))

        if mae_temp < mae_period:
            best_lag=lag
            mae_period=mae_temp
        else:
            print ("mae:" + str(mae_period))
        print ("sale del bucle")

    print ("######best mae is " + str(mae_period) + " with the lag " + str(best_lag))


    model = pf.VAR(df_train,lags=best_lag)
    x = model.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))
    #model.plot_predict_is(h=90, figsize=((8,5)))
    #model.plot_predict(past_values=len(df_train), h=len(df_test), figsize=(8,5))



    future_forecast_pred = model.predict(len(df_test))
    future_forecast_pred = future_forecast_pred[['expected value']]

    list_test = df_test['expected value'].values
    list_future_forecast_pred = future_forecast_pred['expected value'].values

    #pyplot.plot(list_test, label='real')
    #pyplot.plot(list_future_forecast_pred, label='pred')
    #pyplot.legend()
    #pyplot.show()

    #mse_test = (list_future_forecast_pred - list_test)
    #mse_abs_test = abs(mse_test)

    mse = mean_squared_error(list_test, list_future_forecast_pred)
    print('El error medio del modelo_test es: {}'.format(mse))
    rmse = np.sqrt(mse)
    print('El root error medio del modelo_test es: {}'.format(rmse))
    mae = mean_absolute_error(list_test, list_future_forecast_pred)

    df_aler = pd.DataFrame()
    df_aler['real_value'] = list_test
    df_aler['expected value'] = list_future_forecast_pred
    df_aler['mse'] = mse
    df_aler['puntos'] = future_forecast_pred.index
    df_aler.set_index('puntos',inplace=True)
    df_aler['mae'] = mae

    df_aler['anomaly_score'] = abs(df_aler['expected value']-df_aler['real_value'])/df_aler['mae']

    df_aler = df_aler[(df_aler['anomaly_score']> 2)]

    max = df_aler['anomaly_score'].max()
    min = df_aler['anomaly_score'].min()
    df_aler['anomaly_score']= ( df_aler['anomaly_score'] - min ) /(max - min)

    df_aler_ult = df_aler[:5]
    df_aler_ult = df_aler_ult[(df_aler_ult.index==df_aler.index.max())|(df_aler_ult.index==((df_aler.index.max())-1))
                             |(df_aler_ult.index==((df_aler.index.max())-2))|(df_aler_ult.index==((df_aler.index.max())-3))
                             |(df_aler_ult.index==((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = 'FALSE'
    else:
        exists_anom_last_5 = 'TRUE'

    max = df_aler_ult['anomaly_score'].max()
    min = df_aler_ult['anomaly_score'].min()
    df_aler_ult['anomaly_score'] = ( df_aler_ult['anomaly_score'] - min ) /(max - min)
    df_aler_ult = df_aler_ult.fillna(0)
    #####forecast#####

    model_for = pf.VAR(df_var,lags=best_lag)
    x_for = model_for.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))

    # save the model to disk
    filename = "./models_temp/var_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


    future_forecast_pred_for = model_for.predict(num_fut)
    future_forecast_pred_for = future_forecast_pred_for[['expected value']]

    df_result_forecast = future_forecast_pred_for.reset_index()
    df_result_forecast = df_result_forecast.rename(columns = {'index':'step'})



    engine_output={}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.to_dict(orient='record')
    engine_output['engine']='VAR'
    engine_output['future']= df_result_forecast.fillna(0).to_dict(orient='record')

    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='VAR'
    engine_output['future']= df_result_forecast.fillna(0).to_dict(orient='record')
    test_values = pd.DataFrame(future_forecast_pred.values,index = df_test.index,columns=['expected value'])
    test_values['step'] = test_values.index
    engine_output['debug'] = test_values.fillna(0).to_dict(orient='record')

    return (engine_output)
