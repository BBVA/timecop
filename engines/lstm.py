import math
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from .helpers import create_dataset

def anomaly_uni_LSTM(lista_datos, desv_mse=0):
    temp = pd.DataFrame(lista_datos, columns=['values'])
    data_raw = temp.values.astype("float32")

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data_raw)

    print(data_raw)
    TRAIN_SIZE = 0.70

    train_size = int(len(dataset) * TRAIN_SIZE)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Create test and training sets for one-step-ahead regression.
    window_size = 1
    train_X, train_Y = create_dataset(train, window_size)
    test_X, test_Y = create_dataset(test, window_size)
    forecast_X, forecast_Y = create_dataset(dataset, window_size)

    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    forecast_X = np.reshape(
        forecast_X, (forecast_X.shape[0], 1, forecast_X.shape[1]))
    # new engine LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_X, train_Y, epochs=300, batch_size=100, validation_data=(
        test_X, test_Y), verbose=0, shuffle=False)

    yhat = model.predict(test_X)

    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(test_Y.reshape(-1, 1))

    lista_puntos = np.arange(train_size, (train_size + test_size) - 2, 1)
    # print(lista_puntos)
    testing_data = pd.DataFrame(
        yhat_inverse, index=lista_puntos, columns=['expected value'])

    mse = mean_squared_error(testY_inverse, yhat_inverse)
    mae = mean_absolute_error(testY_inverse, yhat_inverse)

    df_aler = pd.DataFrame()
    test = scaler.inverse_transform([test_Y])

    df_aler['real_value'] = test[0]

    df_aler['expected value'] = yhat_inverse
    df_aler['step'] = np.arange(0, len(yhat_inverse), 1)
    df_aler['mae'] = mae
    df_aler['mse'] = mse
    df_aler['anomaly_score'] = abs(
        df_aler['expected value'] - df_aler['real_value']) / df_aler['mae']

    df_aler_ult = df_aler[:5]

    df_aler_ult = df_aler_ult[(df_aler_ult.index == df_aler.index.max()) | (df_aler_ult.index == ((df_aler.index.max())-1))
                              | (df_aler_ult.index == ((df_aler.index.max())-2)) | (df_aler_ult.index == ((df_aler.index.max())-3))
                              | (df_aler_ult.index == ((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = False
    else:
        exists_anom_last_5 = True

    df_aler = df_aler[(df_aler['anomaly_score'] > 2)]

    max = df_aler['anomaly_score'].max()
    min = df_aler['anomaly_score'].min()
    df_aler['anomaly_score'] = (df_aler['anomaly_score'] - min) / (max - min)

    max = df_aler_ult['anomaly_score'].max()
    min = df_aler_ult['anomaly_score'].min()

    df_aler_ult['anomaly_score'] = (
        df_aler_ult['anomaly_score'] - min) / (max - min)

    pred_scaled = model.predict(forecast_X)
    pred = scaler.inverse_transform(pred_scaled)

    print(pred)
    print('prediccion')

    engine_output = {}

    engine_output['rmse'] = str(math.sqrt(mse))
    engine_output['mse'] = int(mse)
    engine_output['mae'] = int(mae)
    engine_output['present_status'] = exists_anom_last_5
    engine_output['present_alerts'] = df_aler_ult.fillna(
        0).to_dict(orient='record')
    engine_output['past'] = df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine'] = 'LSTM'
    df_future = pd.DataFrame(pred[:15], columns=['value'])
    df_future['value'] = df_future.value.astype("float64")
    df_future['step'] = np.arange(len(lista_datos), len(lista_datos)+15, 1)
    engine_output['future'] = df_future.to_dict(orient='record')

    testing_data['expected value'].astype("float64")
    testing_data['step'] = testing_data.index
    testing_data.step.astype("float64")

    engine_output['debug'] = testing_data.to_dict(orient='record')

    return (engine_output)


def anomaly_LSTM(list_var, desv_mse=0):
    print('Training model')
    print(list_var)
    df_var = pd.DataFrame()
    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]
        df_var['var_{}'.format(i)] = list_var[i]

    tam_train = int(len(df_var)*0.7)
    values = df_var.values
    train = values[:tam_train, :]
    print('train'), len(train)

    test = values[tam_train:, :]
    print('test'), len(test)

    train_X, train_y = train[:, :-1], train[:, -1]

    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network

    model.fit(train_X, train_y, epochs=30, batch_size=72, validation_data=(
        test_X, test_y), verbose=0, shuffle=False)

    # make a prediction
    yhat = model.predict(test_X)
    mse = (mean_squared_error(test_y, yhat))
    rmse = np.sqrt(mse)
    print('Model test mean error: {}'.format(rmse))

    yhat = yhat.ravel()

    df_aler = pd.DataFrame()

    df_aler['real_value'] = test_y
    df_aler['expected_value'] = yhat

    df_aler['mse'] = mse
    df_aler['puntos'] = df_aler.index
    df_aler['puntos'] = df_aler['puntos'] + tam_train
    df_aler.set_index('puntos', inplace=True)
    print('paso')

    df_aler['rmse'] = rmse
    mae = mean_absolute_error(yhat, test_y)
    df_aler['mae'] = mean_absolute_error(yhat, test_y)
    print(df_aler.info())

    df_aler['anomaly_score'] = abs(
        df_aler['expected_value']-df_aler['real_value'])/df_aler['mae']
    df_aler_ult = df_aler[:5]
    df_aler = df_aler[(df_aler['anomaly_score'] > 2)]
    max = df_aler['anomaly_score'].max()
    min = df_aler['anomaly_score'].min()

    df_aler['anomaly_score'] = (df_aler['anomaly_score'] - min) / (max - min)

    print('Anomaly')

    df_aler_ult = df_aler[:5]

    df_aler_ult = df_aler_ult[(df_aler_ult.index == df_aler.index.max()) | (df_aler_ult.index == ((df_aler.index.max())-1))
                              | (df_aler_ult.index == ((df_aler.index.max())-2)) | (df_aler_ult.index == ((df_aler.index.max())-3))
                              | (df_aler_ult.index == ((df_aler.index.max())-4))]
    if len(df_aler_ult) == 0:
        exists_anom_last_5 = 'FALSE'
    else:
        exists_anom_last_5 = 'TRUE'
    max = df_aler_ult['anomaly_score'].max()
    min = df_aler_ult['anomaly_score'].min()

    df_aler_ult['anomaly_score'] = (
        df_aler_ult['anomaly_score'] - min) / (max - min)

    # FORECAST PREDICTION
    values = df_var.values
    num_fut = 20

    test1_X, test1_y = values[:, :-1], values[:, -1]

    test1_X = test1_X.reshape((test1_X.shape[0], 1, test1_X.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(test1_X.shape[1], test1_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    model.fit(test1_X, test1_y, epochs=50,
              batch_size=72, verbose=0, shuffle=False)

    len_fore = len(test1_X) - num_fut
    fore = test1_X[len_fore:]
    yhat = model.predict(fore)

    lista_result = np.arange(len(test1_X), (len(test1_X)+num_fut), 1)
    df_result_forecast = pd.DataFrame(
        {'puntos': lista_result, 'expected value': yhat[:, 0]})
    df_result_forecast.set_index('puntos', inplace=True)
    df_result_forecast['expected value'] = df_result_forecast['expected value'].astype(str)
    df_result_forecast['step'] = df_result_forecast.index

    engine_output = {}
    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mae
    engine_output['present_status'] = exists_anom_last_5
    engine_output['present_alerts'] = df_aler_ult.to_dict(orient='record')
    engine_output['past'] = df_aler.to_dict(orient='record')
    engine_output['engine'] = 'LSTM'
    engine_output['future'] = df_result_forecast.to_dict(orient='record')

    return (engine_output)


def forecast_LSTM(list_var, num_fut):
    df_var = pd.DataFrame()
    for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]
        df_var['var_{}'.format(i)] = list_var[i]

    values = df_var.values

    test_X, test_y = values[:, :-1], values[:, -1]

    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(test_X.shape[1], test_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    model.fit(test_X, test_y, epochs=50,
              batch_size=72, verbose=0, shuffle=False)

    len_fore = len(test_X) - num_fut
    fore = test_X[len_fore:]
    yhat = model.predict(fore)

    lista_result = np.arange(len(test_X), (len(test_X) + num_fut), 1)
    df_result = pd.DataFrame({'puntos': lista_result, 'expected value': yhat[:, 0]})
    df_result.set_index('puntos', inplace=True)
    return (df_result)
