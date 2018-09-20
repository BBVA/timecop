import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pyflux as pf


def anomaly_VAR(list_var):
    df_var = pd.DataFrame()

    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]
        df_var['var_{}'.format(i)] = list_var[i]

    tam_train = int(len(df_var)*0.7)
    df_train = df_var[:tam_train]
    print('Train shape: {}'.format(df_train.shape))
    df_test = df_var[tam_train:]
    print('Test shape: {}'.format(df_test.shape))

    model = pf.VAR(df_train, lags=5)
    model.fit()

    future_forecast_pred = model.predict(len(df_test))
    future_forecast_pred = future_forecast_pred[['expected value']]

    list_test = df_test['expected value'].values
    list_future_forecast_pred = future_forecast_pred['expected value'].values

    mse = mean_squared_error(list_test, list_future_forecast_pred)
    print('Model_test mean_error: {}'.format(mse))
    rmse = np.sqrt(mse)
    print('Model_test root error: {}'.format(rmse))
    mae = mean_absolute_error(list_test, list_future_forecast_pred)
    df_aler = pd.DataFrame()

    df_aler['real_value'] = list_test
    df_aler['expected value'] = list_future_forecast_pred
    df_aler['mse'] = mse
    df_aler['puntos'] = future_forecast_pred.index
    df_aler.set_index('puntos', inplace=True)
    df_aler['mae'] = mae

    df_aler['anomaly_score'] = abs(
        df_aler['expected value']-df_aler['real_value'])/df_aler['mae']

    df_aler = df_aler[(df_aler['anomaly_score'] > 2)]

    max = df_aler['anomaly_score'].max()
    min = df_aler['anomaly_score'].min()
    df_aler['anomaly_score'] = (df_aler['anomaly_score'] - min) / (max - min)

    df_aler_ult = df_aler[:5]
    df_aler_ult = df_aler_ult[(df_aler_ult.index == df_aler.index.max()) | (df_aler_ult.index == ((df_aler.index.max())-1))
                              | (df_aler_ult.index == ((df_aler.index.max())-2)) | (df_aler_ult.index == ((df_aler.index.max())-3))
                              | (df_aler_ult.index == ((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = False
    else:
        exists_anom_last_5 = True

    max = df_aler_ult['anomaly_score'].max()
    min = df_aler_ult['anomaly_score'].min()
    print(df_aler_ult)
    df_aler_ult['anomaly_score'] = (
        df_aler_ult['anomaly_score'] - min) / (max - min)

    # forecast

    model_for = pf.VAR(df_var, lags=5)
    model_for.fit()

    future_forecast_pred_for = model_for.predict(5)
    future_forecast_pred_for = future_forecast_pred_for['expected value']
    df_result_forecast = future_forecast_pred_for.reset_index()
    df_result_forecast = df_result_forecast.rename(columns={'index': 'step'})

    print(df_var.head(5))
    print(df_var.tail(5))

    engine_output = {}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status'] = exists_anom_last_5
    engine_output['present_alerts'] = df_aler_ult.to_dict(orient='record')
    engine_output['past'] = df_aler.to_dict(orient='record')
    engine_output['engine'] = 'VAR'
    engine_output['future'] = df_result_forecast.to_dict(orient='record')

    return (engine_output)
