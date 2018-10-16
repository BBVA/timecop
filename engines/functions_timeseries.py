from helpers import merge_two_dicts
from var import anomaly_VAR, univariate_anomaly_VAR
from holtwinter import anomaly_holt
from auto_arima import anomaly_AutoArima
from lstm import anomaly_LSTM, anomaly_uni_LSTM

def model_univariate(lista_datos,num_fut,desv_mse):
    engines_output={}
    debug = {}
    
    try:
        engines_output['LSTM'] = anomaly_uni_LSTM(lista_datos,num_fut,desv_mse)
        debug['LSTM'] = engines_output['LSTM']['debug']
    except Exception as e: 
        print(e)
        print ('ERROR: exception executing LSTM univariate')
       
    try:
        if (len(lista_datos) > 200):
            #new_length= 
            lista_datos=lista_datos[len(lista_datos)-200:]
        engines_output['arima'] = anomaly_AutoArima(lista_datos,num_fut,desv_mse)
        debug['arima'] = engines_output['arima']['debug']
    except  Exception as e: 
        print(e)
        print ('ERROR: exception executing Autoarima')
    
    try:
        engines_output['VAR'] = univariate_anomaly_VAR(lista_datos,num_fut)
        debug['VAR'] = engines_output['VAR']['debug']
    except  Exception as e: 
        print(e)
        print ('ERROR: exception executing VAR')
    
    try:
        engines_output['Holtwinters'] = anomaly_holt(lista_datos,num_fut,desv_mse)
        debug['Holtwinters'] = engines_output['Holtwinters']['debug']
    except  Exception as e: 
        print(e)
        print ('ERROR: exception executing Holtwinters')
    

    best_mae=999999999
    winner='Holtwinters'
    print ('The size is: ')
    print (len(engines_output))
    for key, value in engines_output.iteritems():
        print (key + "   " + str(value['mae']))
        
        if value['mae'] < best_mae:
            best_mae=value['mae']
            winner=key
        print(winner)
            
            
    
        
    print winner
    
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
    except   Exception, err: 
        print(Exception)
        print (err)
        print ('ERROR: exception executing VAR') 

    best_mae=999999999
    winner='LSTM'
    print ('The size is ')
    print (len(engines_output))
    print (debug)
    for key, value in engines_output.iteritems():
        print (key)
        print(str(value['mae']))
        if value['mae'] < best_mae:
            print (key + " " + str(value['mae']) + " best:" + str(best_mae) )
            best_mae=value['mae']
            winner=key
        
    print "el ganador es " + winner
    temp= {}
    temp['debug']=debug
    return merge_two_dicts(engines_output[winner] , temp)
