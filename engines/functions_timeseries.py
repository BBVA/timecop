from engines.helpers import merge_two_dicts
from . var import anomaly_VAR, univariate_anomaly_VAR,univariate_forecast_VAR
from . holtwinter import anomaly_holt,forecast_holt
from . auto_arima import anomaly_AutoArima
from . lstm import anomaly_LSTM, anomaly_uni_LSTM
import traceback

from . BBDD import new_model, get_best_model

from struct import *





def model_univariate(lista_datos,num_fut,desv_mse,train,name):
    engines_output={}
    debug = {}


    if not train:
        # filename = './models_temp/'+name
        # with open(filename,'r') as f:
        #     winner = f.read()
        #     f.close()

        (model_name,model,params)=get_best_model('winner_'+name)
        # print ("recupero el motor " )
        winner= model_name
        if winner == 'LSTM':
            try:
                engines_output['LSTM'] = anomaly_uni_LSTM(lista_datos,num_fut,desv_mse,train,name)
                debug['LSTM'] = engines_output['LSTM']['debug']
            except Exception as e:
                print(e)
                print ('ERROR: exception executing LSTM univariate')
        elif winner == 'VAR':
            engines_output['VAR'] = univariate_forecast_VAR(lista_datos,num_fut,name)
            debug['VAR'] = engines_output['VAR']['debug']
        elif winner == 'Holtwinters':
           engines_output['Holtwinters'] = forecast_holt(lista_datos,num_fut,desv_mse,name)
           debug['Holtwinters'] = engines_output['Holtwinters']['debug']
        else:
            print ("Error")

    else:
        try:
            engines_output['LSTM'] = anomaly_uni_LSTM(lista_datos,num_fut,desv_mse,train,name)
            debug['LSTM'] = engines_output['LSTM']['debug']
        except Exception as e:
            print(e)
            print ('ERROR: exception executing LSTM univariate')

        #try:
            #if (len(lista_datos) > 100):
                ##new_length=
                #lista_datos_ari=lista_datos[len(lista_datos)-100:]
            #engines_output['arima'] = anomaly_AutoArima(lista_datos_ari,num_fut,len(lista_datos),desv_mse)
            #debug['arima'] = engines_output['arima']['debug']
        #except  Exception as e:
            #print(e)
            #print ('ERROR: exception executing Autoarima')

        try:
            if (train):
                engines_output['VAR'] = univariate_anomaly_VAR(lista_datos,num_fut,name)
                debug['VAR'] = engines_output['VAR']['debug']
            else:
                engines_output['VAR'] = univariate_forecast_VAR(lista_datos,num_fut,name)
                debug['VAR'] = engines_output['VAR']['debug']
        except  Exception as e:
            print(e)
            print ('ERROR: exception executing VAR')

        try:
               if (train ):
                   engines_output['Holtwinters'] = anomaly_holt(lista_datos,num_fut,desv_mse,name)
                   debug['Holtwinters'] = engines_output['Holtwinters']['debug']
               else:
                   print ("entra en forecast")
                   engines_output['Holtwinters'] = forecast_holt(lista_datos,num_fut,desv_mse,name)
                   debug['Holtwinters'] = engines_output['Holtwinters']['debug']
        except  Exception as e:
               print(e)
               print ('ERROR: exception executing Holtwinters')


        best_mae=999999999
        winner='VAR'
        print ('The size is: ')
        print (len(engines_output))
        for key, value in engines_output.items():
            print (key + "   " + str(value['mae']))

            if value['mae'] < best_mae:
                best_mae=value['mae']
                winner=key
            print(winner)

        # filename = './models_temp/'+name
        # with open(filename,'w') as f:
        #     f.write(winner)
        #     f.close()
        new_model('winner_'+name, winner, pack('N', 365),'',0)


        print (winner)

    print ("el ganador es " + str(winner))
    print (engines_output[winner])
    temp= {}
    temp['debug']=debug
    return merge_two_dicts(engines_output[winner] , temp)



def model_multivariate(list_var,num_fut,desv_mse):


    engines_output={}
    debug = {}

    try:
        engines_output['LSTM'] = anomaly_LSTM(list_var,num_fut,desv_mse)
        debug['LSTM'] = engines_output['LSTM']['debug']
        print (engines_output['LSTM'])
    except   Exception as e:
        print(e)
        print ('ERROR: exception executing LSTM')

    try:
        engines_output['VAR'] = anomaly_VAR(list_var,num_fut)
        debug['VAR'] = engines_output['VAR']['debug']
        print (engines_output['VAR'])
    except   Exception as e:
        print(Exception)
        print("type error: " + str(e))
        print(traceback.format_exc())
        print ('ERROR: exception executing VAR')

    best_mae=999999999
    winner='LSTM'
    print ('The size is ')
    print (len(engines_output))
    print (debug)
    for key, value in engines_output.items():
        print (key)
        print(str(value['mae']))
        if value['mae'] < best_mae:
            print (key + " " + str(value['mae']) + " best:" + str(best_mae) )
            best_mae=value['mae']
            winner=key

    print ("el ganador es " + winner)
    temp= {}
    temp['debug']=debug
    return merge_two_dicts(engines_output[winner] , temp)
