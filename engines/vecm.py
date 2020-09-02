import numpy as np
from matplotlib import pyplot as plt
import statsmodels.tsa.vector_ar.vecm as vecm
import pandas as pd
from . engine_output_creation import engine_output_creation


def anomaly_vecm(list_var,num_fut=5,desv_mse=2,train=True,name='model-name'):
  df_var = pd.DataFrame()
  for i in range(len(list_var)):
        df_var['var_{}'.format(i)] = list_var[i]

  # split 
  tam_train = int(len(df_var)*0.7)
  #print tam_train
  df_train = df_var[:tam_train]
  print('Tamanio train: {}'.format(df_train.shape))
  df_test = df_var[tam_train:]
  
  lag_order = vecm.select_order(data=df_train, maxlags=10, deterministic="ci", seasons=0)
  rank_test = vecm.select_coint_rank(df_train, 0, 3, method="trace",signif=0.01)
  print ("pasa")
  model = vecm.VECM(df_train, deterministic="ci", seasons=4, coint_rank=rank_test.rank)  # =1
  print ("define")
  vecm_res = model.fit()
  futures = vecm_res.predict(steps=len(df_test))
  # lag_order.summary()
  result=[]
  for list in futures:
    result.append(list[0])

  engine = engine_output_creation('vecm')
  print("empieza")
  df_test['puntos']= df_test.index
  df_test['valores'] = df_test[df_var.columns[0]]

  engine.alerts_creation(result,df_test)
  # # print("empieza")

  engine.metrics_generation(df_test[df_test.columns[0]].values, result)
  # print("empieza")

  engine.debug_creation(result,df_test)


  lag_order = vecm.select_order(data=df_var, maxlags=10, deterministic="ci", seasons=4)
  rank_test = vecm.select_coint_rank(df_var, 0, 3, method="trace",signif=0.01)
  print ("pasa")
  model = vecm.VECM(df_var, deterministic="ci", seasons=4, coint_rank=rank_test.rank)  # =1
  print ("define")
  vecm_res = model.fit()
  futures = vecm_res.predict(steps=num_fut)
  # lag_order.summary()
  result=[]
  for list in futures:
    result.append(list[0])

  engine.forecast_creation( result, df_var.shape[0],num_fut)

  return(engine.engine_output)
        
