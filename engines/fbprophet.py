import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from . helpers import create_train_test
import pickle
from datetime import datetime
import pandas as pd
from fbprophet import Prophet
from . engine_output_creation import engine_output_creation

def anomaly_fbprophet(lista_datos,num_fut,desv_mse=0,train=True,name='model-name'):

    lista_puntos = np.arange(0, len(lista_datos),1)
    df, df_train, df_test = create_train_test(lista_puntos, lista_datos)

    m = Prophet()
    df_temp = pd.DataFrame(df_train['valores'])
    df_temp.rename(columns={"valores": "y"},inplace=True)
    df_temp['ds']= pd.date_range(datetime.today(), periods=len(df_temp['y'])).strftime("%Y/%m/%d").tolist()
    m.fit(df_temp)

    future = m.make_future_dataframe(len(df_test['valores']))
    future.tail()
    forecast = m.predict(future)

    engine = engine_output_creation('fbprophet')
    engine.alerts_creation(forecast[-len(df_test['valores']):]['yhat'],df_test)


        ############## ANOMALY FINISHED,
    print ("Anomaly finished. Start forecasting")
        ############## FORECAST START
    df_temp = pd.DataFrame(df['valores'])
    df_temp.rename(columns={"valores": "y"},inplace=True)
    df_temp['ds']= pd.date_range(datetime.today(), periods=len(df_temp['y'])).strftime("%Y/%m/%d").tolist()
    m_future = Prophet()

    m_future.fit(df_temp)

    future = m_future.make_future_dataframe(num_fut)
    future.tail()
    forecast = m_future.predict(future)

    engine.forecast_creation( forecast[-num_fut:]['yhat'], len(lista_datos),num_fut)
    engine.metrics_generation( df_test['valores'].values, forecast[-len(df_test['valores']):]['yhat'])
    engine.debug_creation(forecast[-len(df['valores']):]['yhat'].tolist(),df_test)

    return (engine.engine_output)
