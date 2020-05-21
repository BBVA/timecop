
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tsa.api import ExponentialSmoothing
from . helpers import create_train_test,seasonal_options
import pickle
from . BBDD import new_model, get_best_model
from struct import *


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(len(seq[int(last):int(last + avg)]))
        last += avg

    return out

def anomaly_holt(lista_datos,num_fut,desv_mse=0,name='NA'):

    lista_puntos = np.arange(0, len(lista_datos),1)


    df, df_train, df_test = create_train_test(lista_puntos, lista_datos)

    engine_output={}

    ####################ENGINE START
    stepwise_model =  ExponentialSmoothing(df_train['valores'],seasonal_periods=1 )
    fit_stepwise_model = stepwise_model.fit()


    fit_forecast_pred_full = fit_stepwise_model.fittedvalues

    future_forecast_pred = fit_stepwise_model.forecast(len(df_test['valores']))


    ###### sliding windows

    #ventanas=h.windows(lista_datos,10)

    #print(ventanas[0])
    #training_data=[]
    #count=0

    #forecast_pred10 =[]
    #real_pred10=[]
    #for slot in ventanas:
        #if count != 0:
            #stepwise_model =  ExponentialSmoothing(training_data,seasonal_periods=1 )
            #fit_stepwise_model = stepwise_model.fit()


            #future_forecast_pred = fit_stepwise_model.forecast(len(slot))
            #forecast_pred10.extend(future_forecast_pred)
            #real_pred10.extend(slot)
            #training_data.extend(slot)

        #else:
            #training_data.extend(slot)
            #forecast_pred10.extend(slot)
            #real_pred10.extend(slot)
            #count=1

    #print ('Wndows prediction')
    ##print ( forecast_pred10)
    ##print ( real_pred10)

    #print ('Wndows mae '  + str(mean_absolute_error(forecast_pred10, real_pred10)))

        ####################ENGINE START

    ##########GRID to find seasonal n_periods
    mae_period = 99999999
    best_period=0
    best_trend='null'
    best_seasonal = 999999
    #list_trend=['add','mul','additive','multiplicative']
    list_trend=['add','mul', 'additive', 'multiplicative'] #,'None']
    print ("pasa hasta aqui")
    periods = seasonal_options(df.valores)
    print (periods)
    #for trend_val in list_trend:
    for seasonal_val in list_trend:
            for period in periods:
                print ('Periodo', period)
                list_forecast_camb = []
                tam_train = int(len(df)*0.7)
                df_test = df[tam_train:]
                part_lenghts = chunkIt(range(len(df_test)),3)

                for i in part_lenghts:
                        print ('Prediccion punto ', i)
                        df_train_camb = df[:tam_train+i]
                        stepwise_model_camb =  ExponentialSmoothing(df_train_camb['valores'] , seasonal=seasonal_val ,seasonal_periods=period ).fit()
                        forecast_camb = stepwise_model_camb.forecast(i)

                        list_forecast_camb.extend(forecast_camb.values[:i])

                mae_temp = mean_absolute_error(list_forecast_camb,df_test['valores'].values)
                if mae_temp < mae_period:
                            best_period = period
    #                        best_trend = trend_val
                            best_seasonal = seasonal_val
                            print ('best_period',best_period)
    #                        print ('best_trend', best_trend)
                            print ('mae_temp', mae_temp)
                            print ('best_seasonal', best_seasonal)
                            mae_period = mae_temp
                else:
                            print ('aa')



    print ("######best mae is " + str(mae_period) + " with the period " + str(best_period)+ " trend " + best_trend)



    stepwise_model =  ExponentialSmoothing(df_train['valores'],seasonal_periods=best_period , seasonal=best_seasonal )
    fit_stepwise_model = stepwise_model.fit()

    future_forecast_pred = fit_stepwise_model.forecast(len(df_test['valores']))
    print (future_forecast_pred.values)



    list_test = df_test['valores'].values
    mse_test = (future_forecast_pred - list_test)
    test_values = pd.DataFrame(future_forecast_pred,index = df_test.index,columns=['expected value'])


    print(list_test)

    mse = mean_squared_error(future_forecast_pred.values,list_test)

    print('Model_test mean error: {}'.format(mse))
    rmse = np.sqrt(mse)
    print('Model_test root error: {}'.format(rmse))

    mse_abs_test = abs(mse_test)

    df_aler = pd.DataFrame(future_forecast_pred,index = df.index,columns=['expected value'])
    df_aler['step'] = df['puntos']
    df_aler['real_value'] = df_test['valores']
    df_aler['mse'] = mse
    df_aler['rmse'] = rmse
    df_aler['mae'] = mean_absolute_error(list_test, future_forecast_pred)
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

    print ("Anomaly finished. Start forecasting")
    stepwise_model1 =  ExponentialSmoothing(df['valores'],seasonal_periods=best_period,seasonal=best_seasonal)
    print ("Pass the training")
    fit_stepwise_model1 = stepwise_model1.fit()


    #with open('./models_temp/learned_model.pkl','w') as f:
    #    pickle.dump(results,f)

    filename='./models_temp/learned_model_holt_winters'+name
    with open(filename,'w') as f:
        f.write(str(best_period)+','+str(best_trend))
        f.close()

    new_model(name, 'Holtwinters', pack('N', 365),str(best_period)+','+str(best_trend),mae_period)


    future_forecast_pred1 = fit_stepwise_model1.forecast(num_fut)
    print ("Pass the forecast")


    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mean_absolute_error(list_test, future_forecast_pred)
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='Holtwinters'
    print ("Only for future")
    df_future= pd.DataFrame(future_forecast_pred1,columns=['value'])
    df_future['value']=df_future.value.astype("float32")
    df_future['step']= np.arange( len(lista_datos),len(lista_datos)+num_fut,1)
    engine_output['future'] = df_future.to_dict(orient='record')
    test_values['step'] = test_values.index
    print ("debug de Holtwinters")
    print (test_values)
    engine_output['debug'] = test_values.to_dict(orient='record')

    print ("la prediccion es")
    print (df_future)

    return engine_output






def forecast_holt(lista_datos,num_fut,desv_mse=0,name='NA'):

    lista_puntos = np.arange(0, len(lista_datos),1)


    df, df_train, df_test = create_train_test(lista_puntos, lista_datos)

    engine_output={}

    best_period=0


    #stepwise_model =  ExponentialSmoothing(df_train['valores'],seasonal_periods=best_period ,trend='add', seasonal='add', )
    #fit_stepwise_model = stepwise_model.fit()
    filename='./models_temp/learned_model_holt_winters'+name
    with open(filename,'r') as f:
        best_period, best_trend= f.read().split(",")
        best_period=int(best_period)
        best_trend=best_trend
        f.close()

    (model_name,model,params)=get_best_model(name)
    print("parametros" + params)
    best_period, best_trend=params.split(",")
    best_period=int(best_period)
    best_trend=best_trend

    print("el dato es ")
    print (str(best_period))
    stepwise_model =  ExponentialSmoothing(df_train['valores'],seasonal_periods=best_period ,trend='add', seasonal='add', )
    fit_stepwise_model = stepwise_model.fit()


    future_forecast_pred = fit_stepwise_model.forecast(len(df_test['valores']))
    print (future_forecast_pred.values)



    list_test = df_test['valores'].values
    mse_test = (future_forecast_pred - list_test)
    test_values = pd.DataFrame(future_forecast_pred,index = df_test.index,columns=['expected value'])


    print(list_test)

    mse = mean_squared_error(future_forecast_pred.values,list_test)

    print('Model_test mean error: {}'.format(mse))
    rmse = np.sqrt(mse)
    print('Model_test root error: {}'.format(rmse))

    mse_abs_test = abs(mse_test)

    df_aler = pd.DataFrame(future_forecast_pred,index = df.index,columns=['expected value'])
    df_aler['step'] = df['puntos']
    df_aler['real_value'] = df_test['valores']
    df_aler['mse'] = mse
    df_aler['rmse'] = rmse
    df_aler['mae'] = mean_absolute_error(list_test, future_forecast_pred)
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

    print ("Anomaly finished. Start forecasting")
    stepwise_model1 =  ExponentialSmoothing(df['valores'],seasonal_periods=best_period , seasonal='add')
    print ("Pass the training")
    fit_stepwise_model1 = stepwise_model1.fit()




    future_forecast_pred1 = fit_stepwise_model1.forecast(num_fut)
    print ("Pass the forecast")


    engine_output['rmse'] = rmse
    engine_output['mse'] = mse
    engine_output['mae'] = mean_absolute_error(list_test, future_forecast_pred)
    engine_output['present_status']=exists_anom_last_5
    engine_output['present_alerts']=df_aler_ult.fillna(0).to_dict(orient='record')
    engine_output['past']=df_aler.fillna(0).to_dict(orient='record')
    engine_output['engine']='Holtwinters'
    print ("Only for future")
    df_future= pd.DataFrame(future_forecast_pred1,columns=['value'])
    df_future['value']=df_future.value.astype("float32")
    df_future['step']= np.arange( len(lista_datos),len(lista_datos)+num_fut,1)
    engine_output['future'] = df_future.to_dict(orient='record')
    test_values['step'] = test_values.index
    print ("debug de Holtwinters")
    print (test_values)
    engine_output['debug'] = test_values.to_dict(orient='record')

    print ("la prediccion es")
    print (df_future)

    return engine_output

#def forecast_holt(lista_datos, num_fut):

    #lista_puntos = np.arange(0, len(lista_datos),1)

    #df = pd.DataFrame()
    #df['puntos'] = lista_puntos
    #df['valores'] = lista_datos

    #df.set_index('puntos',inplace=True)

    #stepwise_model =  ExponentialSmoothing(df,seasonal_periods=len(df) , seasonal='add')
    #fit_stepwise_model = stepwise_model.fit()

    #fit_forecast_pred = fit_stepwise_model.fittedvalues

    #future_forecast_pred = fit_stepwise_model.forecast(num_fut)

    #df_result = pd.DataFrame({'puntos':future_forecast_pred.index, 'valores':future_forecast_pred.values})
    #df_result.set_index('puntos',inplace=True)
    #return (df_result)
