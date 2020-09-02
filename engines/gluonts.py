import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from . helpers import create_train_test
import pickle
from datetime import datetime
import pandas as pd
from . engine_output_creation import engine_output_creation
from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset



def anomaly_gluonts(lista_datos,num_fut,desv_mse=0,train=True,name='model-name'):
    lista_puntos = np.arange(0, len(lista_datos),1)
    df, df_train, df_test = create_train_test(lista_puntos, lista_datos)


    data_list = [{"start": "01-01-2012 04:05:00", "target": df_train['valores'].values}]

    dataset = ListDataset(
        data_iter=data_list,
        freq="5min"
    )

    trainer = Trainer(epochs=15)
    estimator = deepar.DeepAREstimator(freq="5min", prediction_length=len(df_test['valores']), trainer=trainer)
    predictor = estimator.train(training_data=dataset)

    prediction = next(predictor.predict(dataset))

    engine = engine_output_creation('gluonts')
    engine.alerts_creation(prediction.mean.tolist(),df_test)
    engine.debug_creation(prediction.mean.tolist(),df_test)
    print ( 'longitud del test' + str(df_test.shape) + 'frente a la prediccion' + str(len(prediction.mean.tolist())))
    engine.metrics_generation( df_test['valores'].values,prediction.mean.tolist())


        ############## ANOMALY FINISHED,
    print ("Anomaly finished. Start forecasting")
        ############## FORECAST START


    data_list = [{"start": "01-01-2012 04:05:00", "target": df['valores'].values}]

    dataset = ListDataset(
        data_iter=data_list,
        freq="5min"
    )

    trainer = Trainer(epochs=15)
    estimator = deepar.DeepAREstimator(freq="5min", prediction_length=num_fut, trainer=trainer)
    predictor = estimator.train(training_data=dataset)

    prediction = next(predictor.predict(dataset))


    engine.forecast_creation( prediction.mean.tolist(), len(lista_datos),num_fut)
    return (engine.engine_output)
