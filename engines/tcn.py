import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from tcn import TCN

from . helpers import create_train_test
from . engine_output_creation import engine_output_creation



def anomaly_tcn(lista_datos,num_fut,desv_mse=0,train=True,name='model-name'):
    lookback_window = 15
    # lista_puntos = np.arange(lookback_window-1, len(lista_datos)+ lookback_window -1,1)
    # df, df_train, df_test = create_train_test(lista_puntos, lista_datos)



    x, y = [], []
    for i in range(lookback_window, len(lista_datos)):
        x.append(lista_datos[i - lookback_window:i])
        y.append(lista_datos[i])
    x = np.array(x)
    y = np.array(y)

    train_part = round(len(x) * 0.7)
    print(train_part)
    x_train = x[:train_part]
    y_train = y[:train_part]
    x_test = x[train_part:]
    y_test = y[train_part:]

    print(x.shape)
    print(y.shape)

    i = Input(shape=(lookback_window, 1))
    m = TCN()(i)
    m = Dense(1, activation='linear')(m)

    model = Model(inputs=[i], outputs=[m])

    model.summary()

    model.compile('adam', 'mae')

    print('Train...')
    model.fit(x_train, y_train, epochs=10, verbose=2)

    p = model.predict(x_test)


    df_test = pd.DataFrame()
    df_test['valores'] = lista_datos[train_part+lookback_window:]
    df_test['puntos'] = np.arange(train_part+lookback_window-1, len(lista_datos)-1)
    df_test = df_test.set_index('puntos')
    df_test['puntos'] = df_test.index

    engine = engine_output_creation('tcn')
    engine.alerts_creation(p.reshape(len(y_test)),df_test)
    engine.debug_creation(p.tolist(),df_test)
    engine.metrics_generation( df_test['valores'].values, p)




        ############## ANOMALY FINISHED,
    print ("Anomaly finished. Start forecasting")


    i = Input(shape=(lookback_window, 1))
    m = TCN()(i)
    m = Dense(1, activation='linear')(m)

    model2 = Model(inputs=[i], outputs=[m])

    model2.summary()

    model2.compile('adam', 'mae')

    print('Train...')
    model2.fit(x, y, epochs=10, verbose=2)

    p_2 = model2.predict(x[-num_fut:])

    print("predicted")
    engine.forecast_creation( p_2.tolist(), len(lista_datos),num_fut)

    return (engine.engine_output)
