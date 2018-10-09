import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pyflux as pf
import helpers as h




def univariate_anomaly_VAR(lista_datos):
    lista_puntos = np.arange(0, len(lista_datos),1)


    df = pd.DataFrame()
    df['valores'] = lista_datos

    
    tam_train = int(len(df)*0.7)
    #print tam_train
    df_train = df[:tam_train]
    print('Tamanio train: {}'.format(df_train.shape))
    df_test = df[tam_train:]
    print('Tamanio test: {}'.format(df_test.shape))

    model = pf.VAR(df_train,lags=15)
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
    print df_aler_ult
    df_aler_ult['anomaly_score'] = ( df_aler_ult['anomaly_score'] - min ) /(max - min)
    
    #####forecast#####
    
    model_for = pf.VAR(df,lags=5)
    x_for = model_for.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))
    
    future_forecast_pred_for = model_for.predict(5)
    
    df_result_forecast = future_forecast_pred_for.reset_index()
    df_result_forecast = df_result_forecast.rename(columns = {'index':'step'})

    print df.head(5)
    print df.tail(5)
    
    engine_output={}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.to_dict(orient='record')
    engine_output['past']=df_aler.to_dict(orient='record')
    engine_output['engine']='VAR'
    engine_output['future']= df_result_forecast.to_dict(orient='record')
    test_values = pd.DataFrame(future_forecast_pred.values,index = df_test.index,columns=['expected value'])
    test_values['step'] = test_values.index
    engine_output['debug'] = test_values.to_dict(orient='record')
    
    
    return (engine_output)


def anomaly_VAR(list_var):
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
    
    lags = int(round(len(df_test)/2))
    model = pf.VAR(df_train,lags=lags)
    x = model.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))
    #model.plot_predict_is(h=90, figsize=((8,5)))
    #model.plot_predict(past_values=len(df_train), h=len(df_test), figsize=(8,5))
    

    future_forecast_pred = model.predict(len(df_test))
    future_forecast_pred = future_forecast_pred[['expected value']]
    
    list_test = df_test['expected value'].values
    list_future_forecast_pred = future_forecast_pred['expected value'].values
    
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
    
    model_for = pf.VAR(df_var,lags=5)
    x_for = model_for.fit()

    #model.plot_z(list(range(0,6)),figsize=(15,5))
    #model.plot_fit(figsize=(8,5))
    
    future_forecast_pred_for = model_for.predict(5)
    future_forecast_pred_for = future_forecast_pred_for[['expected value']]
    
    df_result_forecast = future_forecast_pred_for.reset_index()
    df_result_forecast = df_result_forecast.rename(columns = {'index':'step'})


    
    engine_output={}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.to_dict(orient='record')
    engine_output['past']=df_aler.to_dict(orient='record')
    engine_output['engine']='VAR'
    engine_output['future']= df_result_forecast.to_dict(orient='record')

    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.to_dict(orient='record')
    engine_output['past']=df_aler.to_dict(orient='record')
    engine_output['engine']='VAR'
    engine_output['future']= df_result_forecast.to_dict(orient='record')
    test_values = pd.DataFrame(future_forecast_pred.values,index = df_test.index,columns=['expected value'])
    test_values['step'] = test_values.index
    engine_output['debug'] = test_values.to_dict(orient='record')
    
    return (engine_output)
