
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential

import math
#import helpers as h
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error

from tbats import BATS, TBATS


from numpy.random import seed
seed(69)
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle

#import multiprocessing
from . BBDD import new_model, get_best_model
from . helpers import create_train_test,seasonal_options









def anomaly_uni_TBATS(lista_datos,num_forecast=10,desv_mse=2,train='True',name='test'):

    lista_puntos = np.arange(0, len(lista_datos),1)

    df, df_train, df_test = create_train_test(lista_puntos, lista_datos)

    engine_output={}

    actual_model=''

    if (train):

        ##########################################################################################
        #############################################################################################3
        periods = seasonal_options(df.valores)
        estimator = TBATS(seasonal_periods= periods[:2])
        # Fit model
        print("Starting Anomaly Model Fitted")

        fitted_model = estimator.fit(df_train['valores'])
        print("Anomaly Model Fitted")

        # Forecast 14 steps ahead
        anomaly_forecasted = fitted_model.forecast(steps=len(df_test['valores']))


        mae = mean_absolute_error(anomaly_forecasted,df_test['valores'].values)



        #mae = mean_absolute_error(y_forecasted,df_test['valores'].values)

        df_aler = pd.DataFrame(anomaly_forecasted,index = df_test.index,columns=['expected value'])
        df_aler['step'] = df['puntos']
        df_aler['real_value'] = df_test['valores']
        df_aler['mae'] = mean_absolute_error(anomaly_forecasted, df_test['valores'].values)
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

        # Fit model
        fitted_model = estimator.fit(df['valores'])
        print("Forecast Model Fitted")

        # Forecast num_forecast steps ahead
        y_forecasted = fitted_model.forecast(steps=num_forecast)

        df_future= pd.DataFrame(y_forecasted,columns=['value'])
        df_future['value']=df_future.value.astype("float32")
        df_future['step']= np.arange( len(lista_datos),len(lista_datos)+num_forecast,1)



        #engine_output['rmse'] = rmse
        #engine_output['mse'] = mse
        engine_output['mae'] = mae
        engine_output['present_status']=exists_anom_last_5
        engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
        engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
        engine_output['engine']='TBATS'
        print ("Only for future")


        engine_output['future'] = df_future.to_dict(orient='record')
        test_values = pd.DataFrame(anomaly_forecasted,index = df_test.index,columns=['expected value'])

        test_values['step'] = test_values.index
        #print ("debug de Holtwinters")
        #print (test_values)
        engine_output['debug'] = test_values.to_dict(orient='record')

        #print ("la prediccion es")
        #print (df_future)

        return engine_output
