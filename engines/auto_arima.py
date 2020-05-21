import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from . engine_output_creation import engine_output_creation
from . helpers import create_train_test
import pickle

def anomaly_AutoArima(lista_datos,num_fut,desv_mse=0,train='True',name='model-name'):
    print("Starting Autorima engine")
    orig_size= len(lista_datos)
    if (len(lista_datos) > 100):
        lista_datos_orig=lista_datos
        lista_datos=lista_datos[len(lista_datos)-100:]
    else:
        lista_datos_orig=lista_datos

    if orig_size < 100:
        start_point =0
    else:
        start_point= int(orig_size) - 100
    lista_puntos = np.arange(start_point, orig_size,1)

    df, df_train, df_test = create_train_test(lista_puntos, lista_datos)

    engine_output={}
    stepwise_model =  pm.auto_arima(df_train['valores'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                              start_P=0, seasonal=True, d=1, D=1, trace=False, approx=False,
                              error_action='ignore',  # don't want to know if an order does not work
                              suppress_warnings=True,  # don't want convergence warnings
                              c=False,
                              disp=-1,
                              stepwise=True)  # set to stepwise

    print ("Fitted first model")
    stepwise_model.fit(df_train['valores'])

    fit_forecast_pred = stepwise_model.predict_in_sample(df_train['valores'])
    fit_forecast = pd.DataFrame(fit_forecast_pred,index = df_train.index,columns=['Prediction'])

    future_forecast_pred = stepwise_model.predict(n_periods=len(df_test['valores']))
    future_forecast = pd.DataFrame(future_forecast_pred,index = df_test.index,columns=['Prediction'])
    print(df_test.index)

    list_test = df_test['valores'].values
    mse_test = (future_forecast_pred - list_test)

    engine = engine_output_creation('Autoarima')
    engine.alerts_creation(future_forecast_pred,df_test)
    engine.debug_creation(future_forecast_pred,df_test)
    engine.metrics_generation( df_test['valores'].values,future_forecast_pred)

    ############## ANOMALY FINISHED,
    print ("Anomaly finished. Start forecasting")

    ############## FORECAST START
    updated_model = stepwise_model.fit(df['valores'])

    filename = './arima_' +name
    # Serialize with Pickle
    with open(filename, 'wb') as pkl:
        pickle.dump(stepwise_model, pkl)
    forecast = updated_model.predict(n_periods=num_fut)

    engine.forecast_creation( forecast,orig_size,num_fut)
    print("Ended Autorima engine")

    return (engine.engine_output)
