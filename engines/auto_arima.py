import numpy as np
import pandas as pd
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error,mean_absolute_error
import helpers as h

def anomaly_AutoArima(lista_datos,num_fut,desv_mse=0):
    
    lista_puntos = np.arange(0, len(lista_datos),1)

    df, df_train, df_test = h.create_train_test(lista_puntos, lista_datos) 
    
    engine_output={}
    
    stepwise_model = auto_arima(df_train['valores'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                              start_P=0, seasonal=True, d=1, D=1, trace=False, approx=False,
                              error_action='ignore',  # don't want to know if an order does not work
                              suppress_warnings=True,  # don't want convergence warnings
                              c=False,
                              disp=-1,
                              stepwise=False)  # set to stepwise

    stepwise_model.fit(df_train['valores'])

    fit_forecast_pred = stepwise_model.predict_in_sample(df_train['valores'])
    fit_forecast = pd.DataFrame(fit_forecast_pred,index = df_train.index,columns=['Prediction'])
    
    future_forecast_pred = stepwise_model.predict(n_periods=len(df_test['valores']))
    future_forecast = pd.DataFrame(future_forecast_pred,index = df_test.index,columns=['Prediction'])

    list_test = df_test['valores'].values
    mse_test = (future_forecast_pred - list_test)

    mse = mean_squared_error(list_test, future_forecast_pred)
    rmse = np.sqrt(mse)
    mse_abs_test = abs(mse_test)
    
    df_aler = pd.DataFrame(future_forecast_pred,index = df_test.index,columns=['expected value'])
    df_aler['step'] = df_test['puntos']
    df_aler['real_value'] = df_test['valores']
    df_aler['mse'] = mse
    df_aler['rmse'] = rmse
    df_aler['mae'] = mean_absolute_error(list_test, future_forecast_pred)
    df_aler['anomaly_score'] = abs(df_aler['expected value'] - df_aler['real_value']) / df_aler['mae']

    print ('Last alerts')

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


    ############## ANOMALY FINISHED, 
    print ("Anomaly finished. Start forecasting")

    ############## FORECAST START
    updated_model = stepwise_model.fit(df['valores'])
    
    forecast = updated_model.predict(n_periods=num_fut)
    
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mean_absolute_error(list_test, future_forecast_pred)
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='Autoarima'
    df_future= pd.DataFrame(forecast,columns=['value'])
    df_future['value']=df_future.value.astype("float32")
    df_future['step']= np.arange( len(lista_datos),len(lista_datos)+num_fut,1)
    engine_output['future'] = df_future.to_dict(orient='record')
    testing_data  = pd.DataFrame(future_forecast_pred,index = df_test.index,columns=['expected value'])
    testing_data['step']=testing_data.index
    engine_output['debug'] = testing_data.to_dict(orient='record')
    
    return (engine_output)
