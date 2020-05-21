import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from . helpers import create_train_test
import pickle
from datetime import datetime
import pandas as pd
from fbprophet import Prophet


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


class engine_output_creation:

  def __init__(self, engine_name):
    self.engine_name = engine_name
    self.engine_output={}
    self.engine_output['engine']=engine_name


  def alerts_creation(self,forecasted_list , df_test):

    df_aler = pd.DataFrame(df_test['puntos'],index = df_test.index,columns=['expected value'])
    df_aler.rename(columns={df_test.columns[0]: "step"},inplace=True)
    df_aler['expected value'] = forecasted_list

    df_aler['real_value'] = df_test['valores']
    list_test = df_test['valores'].values

    mse = mean_squared_error(list_test, forecasted_list)
    rmse = np.sqrt(mse)
    df_aler['step'] = df_test['puntos']

    df_aler['mse'] = mse
    df_aler['rmse'] = rmse

    df_aler['mae'] = mean_absolute_error(list_test, forecasted_list)
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
    self.engine_output['present_status']=exists_anom_last_5
    self.engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    self.engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    return('ok')

  def forecast_creation(self,forecasted_list , start_step,num_fut):
    df_future= pd.DataFrame(forecasted_list.tolist(),columns=['value'])
    df_future.rename(columns={df_future.columns[0]: "value"},inplace=True)

    df_future['value']=df_future.value.astype("float32")

    df_future['step']= np.arange( start_step,start_step+num_fut,1)
    self.engine_output['future'] = df_future.to_dict(orient='record')
    return('OK')



  def debug_creation (self, trained_data, df_test):
    testing_data  = pd.DataFrame(trained_data,index = df_test.index,columns=['value']) #,columns=['expected value'])
    testing_data.rename(columns={"yhat": "expected value"},inplace=True)

    testing_data['step']=testing_data.index
    self.engine_output['debug'] = testing_data.to_dict(orient='record')

  def metrics_generation(self, list_test, list_yhat):
    #list_test = df_test['valores'].values
    mse = mean_squared_error(list_test, list_yhat)
    rmse = np.sqrt(mse)
    self.engine_output['rmse'] = rmse
    self.engine_output['mse'] = mse
    self.engine_output['mae'] = mean_absolute_error(list_test, list_yhat)
    self.engine_output['smape'] = smape(list_test, list_yhat)
