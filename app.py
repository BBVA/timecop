from flask import request, Flask, jsonify, abort
from flask_cors import CORS
import json

import engines.functions_timeseries as ft
import engines.BBDD as db
import os
from celery import Celery



# import engines functions_timeseries
from engines.helpers import merge_two_dicts,trendline
from engines.var import anomaly_VAR, univariate_anomaly_VAR,univariate_forecast_VAR,anomaly_var
from engines.holtwinter import anomaly_holt,forecast_holt
from engines.auto_arima import anomaly_AutoArima
from engines.lstm import anomaly_LSTM, anomaly_uni_LSTM
from engines.fbprophet import anomaly_fbprophet
from engines.gluonts import anomaly_gluonts
from engines.nbeats import anomaly_nbeats
from engines.vecm import anomaly_vecm
from engines.tcn import anomaly_tcn

from engines.changepointdetection import find_changepoints


from datetime import datetime



from struct import *


app = Flask(__name__)
CORS(app)


app.config.from_pyfile(os.path.join(".", "config/app.cfg"), silent=False)

db.init_database()

DB_NAME= app.config.get("DB_NAME")
PORT = app.config.get("PORT")

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
#broker=main.config['CELERY_BROKER_URL'])

celery.conf.update(app.config)


@app.route('/univariate', methods=['POST'])
def univariate_engine():
    db.init_database()

    if not request.json:
        abort(400)


    timedata = request.get_json()
    print (timedata)
    lista=timedata['data']

    num_fut = int(timedata.get('num_future', 5))
    desv_mae = int(timedata.get('desv_metric', 2))
    name = timedata.get('name', 'NA')
    train = timedata.get('train', True)
    restart = timedata.get('restart', False)

    print ("train?"+ str(train))
    print ("restart?" + str(restart))
    print ("Received TS")


    if(name != 'NA'):
        filename= './lst/'+name+'.lst'
        try:
            # with open(filename, 'r') as filehandle:
            #     previousList = json.load(filehandle)
            previousList=db.get_ts(name).split(',')
            previousList = list(map(int, previousList))
        except Exception:
            previousList=[]
        print ("previous list" )

        if  not restart :
            print ("Lista append")
            lista = previousList + lista
        # with open(filename, 'w') as filehandle:
        #     json.dump(lista,filehandle)
        str_lista= ",".join(str(v) for v in lista)
        db.set_ts(name,str_lista)

    #desv_mse = 0
    print ("la lista al final es "+ str(type(lista)))
    print (lista)

    salida = ft.model_univariate(lista,num_fut,desv_mae,train,name)

    return jsonify(salida), 201


@app.route('/back_univariate', methods=['POST'])
def back_univariate_engine():
    db.init_database()

    if not request.json:
        abort(400)

    timedata = request.get_json()
    print (timedata)
    lista=timedata['data']

    num_fut = int(timedata.get('num_future', 5))
    desv_mae = int(timedata.get('desv_metric', 2))
    name = timedata.get('name', 'NA')
    train = timedata.get('train', True)
    restart = timedata.get('restart', False)

    print ("train?"+ str(train))
    print ("restart?" + str(restart))
    print ("Received TS")


    if(name != 'NA'):
        filename= './lst/'+name+'.lst'
        try:
            # with open(filename, 'r') as filehandle:
            #     previousList = json.load(filehandle)
            previousList=db.get_ts(name).split(',')
            previousList = list(map(int, previousList))
        except Exception:
            previousList=[]
        print ("previous list" )

        if  not restart :
            print ("Lista append")
            lista = previousList + lista
        # with open(filename, 'w') as filehandle:
        #     json.dump(lista,filehandle)
        str_lista= ",".join(str(v) for v in lista)
        db.set_ts(name,str_lista)

    #desv_mse = 0
    print ("la lista al final es "+ str(type(lista)))
    print (lista)
    print (name )

    print ("invoco el backend")
    salida = back_model_univariate.s(lista_datos=lista,num_fut=num_fut,desv_mse=desv_mae,train=train,name=name).apply_async()

    print (salida.id)

    #task = long_task.apply_async()
    valor = {'task_id': salida.id}
    return jsonify(valor), 200
    #return jsonify(salida), 201

@app.route('/back_univariate_status/<task_id>')
def univariate_taskstatus(task_id):
    task = back_model_univariate.AsyncResult(task_id)
    print ("llega aqui")
    print (task)
    print("----"+task.state+"----")

    if task.state == 'PENDING':
        response = {
            'state': 'Pending',
            'current': 0,
            'total': 1,
            'status': 'Pending...',
            'result': 'Pending'
        }
    if task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            'result': task.info.get('result', ''),
            'response': task.info
        }
    if task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'current': 6,
            'total': 6,
            'result': task.info.get('result', ''),
            'status': task.info.get('status', 'Sucessfully'),
            'task_dump': str(task)
        }
        # if 'result' in task.info:
        #     print ("el result aparece en el SUCCESS")
        #     response['result'] = task.info['result']
        # else:
        #     print ("el result NO aparece en el SUCCESS")


    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            'result': task.info.get('result', ''),
            'response': task.info
        }
    print (task.state)
    print(task.info)
    return jsonify(response)




############################backen functions


@celery.task(bind=True)
def back_model_univariate(self, lista_datos,num_fut,desv_mse,train,name):
    engines = {'LSTM': 'anomaly_uni_LSTM(lista_datos,num_fut,desv_mse,train,name)',
                'VAR': 'anomaly_var(lista_datos,num_fut,desv_mse,train,name)',
                'nbeats': 'anomaly_nbeats(lista_datos,num_fut,desv_mse,train,name)',
                'gluonts': 'anomaly_gluonts(lista_datos,num_fut,desv_mse,train,name)',
                'fbprophet': 'anomaly_fbprophet(lista_datos,num_fut,desv_mse,train,name)',
                'arima': 'anomaly_AutoArima(lista_datos,num_fut,desv_mse,train,name)',
                'Holtwinters': 'anomaly_holt(lista_datos_holt,num_fut,desv_mse,name)',
                'tcn': 'anomaly_tcn(lista_datos,num_fut,desv_mse,train,name)'
                }

    engines_output={}
    debug = {}

    temp_info = {}

    starttime = datetime.now()


    # Holtwinters workaround, must to solve
    if (len(lista_datos) > 2000):

        lista_datos_holt=lista_datos[len(lista_datos)-2000:]
    else:
        lista_datos_holt = lista_datos

    counter = 0
    for engine_name, engine_function in engines.items() :
        self.update_state(state='PROGRESS',
                          meta={'running': engine_name,
                                'status': temp_info,
                                'total': len(engines),
                                'finish': counter })
        try:
            engines_output[engine_name] = eval(engine_function)
            debug[engine_name] = engines_output[engine_name]['debug']
            temp_info[engine_name]=engines_output[engine_name]
            self.update_state(state='PROGRESS',
                      meta={'running': engine_name,
                            'status': temp_info,
                            'total': 6,
                            'finish': 1})
        except Exception as e:

            print ('ERROR: ' + engine_name +' univariate: ' + str(e) )
        counter = counter + 1

        if (train):

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

            db.new_model('winner_'+name, winner, pack('N', 365),'',0)


            print (winner)











    # self.update_state(state='PROGRESS',
    #                   meta={'running': 'LSTM',
    #                         'status': '',
    #                         'total': 6,
    #                         'finish': 0 })
    # if not train:
    #
    #     (model_name,model,params)=db.get_best_model('winner_'+name)
    #     # print ("recupero el motor " )
    #     winner= model_name
    #     if winner == 'LSTM':
    #         try:
    #             engines_output['LSTM'] = anomaly_uni_LSTM(lista_datos,num_fut,desv_mse,train,name)
    #             debug['LSTM'] = engines_output['LSTM']['debug']
    #         except Exception as e:
    #             print(e)
    #             print ('ERROR: exception executing LSTM univariate')
    #     elif winner == 'VAR':
    #         engines_output['VAR'] = univariate_forecast_VAR(lista_datos,num_fut,name)
    #         debug['VAR'] = engines_output['VAR']['debug']
    #     elif winner == 'Holtwinters':
    #        engines_output['Holtwinters'] = forecast_holt(lista_datos,num_fut,desv_mse,name)
    #        debug['Holtwinters'] = engines_output['Holtwinters']['debug']
    #     else:
    #         print ("Error")
    #
    # else:
    #
    #
    #
    #
    #     try:
    #         engines_output['nbeats'] = anomaly_nbeats(lista_datos,num_fut,desv_mse,train,name)
    #         debug['nbeats'] = engines_output['nbeats']['debug']
    #         temp_info['nbeats']=engines_output['nbeats']
    #         self.update_state(state='PROGRESS',
    #                   meta={'running': 'nbeats',
    #                         'status': temp_info,
    #                         'total': 7,
    #                         'finish': 1})
    #     except Exception as e:
    #
    #         print ('ERROR: nbeats univariate: ' + str(e) )
    #
    #
    #
    #     try:
    #         engines_output['gluonts'] = anomaly_gluonts(lista_datos,num_fut,desv_mse,train,name)
    #         debug['gluonts'] = engines_output['gluonts']['debug']
    #         temp_info['gluonts']=engines_output['gluonts']
    #         self.update_state(state='PROGRESS',
    #                   meta={'running': 'gluonts',
    #                         'status': temp_info,
    #                         'total': 6,
    #                         'finish': 1})
    #     except Exception as e:
    #
    #         print ('ERROR: gluonts univariate: ' + str(e) )
    #
    #
    #     try:
    #         engines_output['fbprophet'] = anomaly_fbprophet(lista_datos,num_fut,desv_mse,train,name)
    #         debug['fbprophet'] = engines_output['fbprophet']['debug']
    #         temp_info['fbprophet']=engines_output['fbprophet']
    #         self.update_state(state='PROGRESS',
    #                   meta={'running': 'fbprophet',
    #                         'status': temp_info,
    #                         'total': 6,
    #                         'finish': 2})
    #     except Exception as e:
    #
    #         print ('ERROR: fbprophet univariate: ' + str(e) )
    #
    #
    #     try:
    #
    #         engines_output['arima'] = anomaly_AutoArima(lista_datos,num_fut,desv_mse,train,name)
    #         debug['arima'] = engines_output['arima']['debug']
    #         temp_info['arima']=engines_output['arima']
    #         self.update_state(state='PROGRESS',
    #                   meta={'running': 'VAR',
    #                         'status': temp_info,
    #                         'total': 6,
    #                         'finish': 3})
    #     except  Exception as e:
    #         print(e)
    #         print ('ERROR: exception executing Autoarima')
    #
    #     try:
    #         if (train):
    #             engines_output['VAR'] = univariate_anomaly_VAR(lista_datos,num_fut,name)
    #             debug['VAR'] = engines_output['VAR']['debug']
    #         else:
    #             engines_output['VAR'] = univariate_forecast_VAR(lista_datos,num_fut,name)
    #             debug['VAR'] = engines_output['VAR']['debug']
    #         temp_info['VAR'] = engines_output['VAR']
    #         self.update_state(state='PROGRESS',
    #                   meta={'running': 'Holtwinters',
    #                         'status': temp_info,
    #                         'total': 6,
    #                         'finish': 4})
    #
    #     except  Exception as e:
    #         print(e)
    #         print ('ERROR: exception executing VAR')
    #
    #     try:
    #         if (train ):
    #                 if (len(lista_datos) > 2000):
    #                     #new_length=
    #                     lista_datos_holt=lista_datos[len(lista_datos)-2000:]
    #                 else:
    #                     lista_datos_holt = lista_datos
    #                 engines_output['Holtwinters'] = anomaly_holt(lista_datos_holt,num_fut,desv_mse,name)
    #                 debug['Holtwinters'] = engines_output['Holtwinters']['debug']
    #         else:
    #                print ("entra en forecast")
    #                if (len(lista_datos) > 2000):
    #                    #new_length=
    #                    lista_datos_holt=lista_datos[len(lista_datos)-2000:]
    #                else:
    #                    lista_datos_holt = lista_datos
    #                engines_output['Holtwinters'] = forecast_holt(lista_datos,num_fut,desv_mse,name)
    #                debug['Holtwinters'] = engines_output['Holtwinters']['debug']
    #
    #         temp_info['Holtwinters'] = engines_output['Holtwinters']
    #         self.update_state(state='PROGRESS',
    #                   meta={'running': 'Holtwinters',
    #                         'status': temp_info,
    #                         'total': 6,
    #                         'finish': 5})
    #
    #     except  Exception as e:
    #            print(e)
    #            print ('ERROR: exception executing Holtwinters')
    #
    #     try:
    #         engines_output['LSTM'] = anomaly_uni_LSTM(lista_datos,num_fut,desv_mse,train,name)
    #         debug['LSTM'] = engines_output['LSTM']['debug']
    #         temp_info['LSTM']=engines_output['LSTM']
    #         self.update_state(state='PROGRESS',
    #                   meta={'running': 'anomaly_AutoArima',
    #                         'status': temp_info,
    #                         'total': 6,
    #                         'finish': 6})
    #     except Exception as e:
    #         print(e)
    #         print ('ERROR: exception executing LSTM univariate')
    #
    #
    #     best_mae=999999999
    #     winner='VAR'
    #     print ('The size is: ')
    #     print (len(engines_output))
    #     for key, value in engines_output.items():
    #         print (key + "   " + str(value['mae']))
    #
    #         if value['mae'] < best_mae:
    #             best_mae=value['mae']
    #             winner=key
    #         print(winner)
    #
    #     db.new_model('winner_'+name, winner, pack('N', 365),'',0)
    #
    #
    #     print (winner)

    print ("el ganador es " + str(winner))
    print (engines_output[winner])
    temp= {}
    temp['debug']=debug
    temp['trend']= trendline(lista_datos)

#    return merge_two_dicts(engines_output[winner] , temp)
    salida = merge_two_dicts(engines_output[winner], temp_info)
    finishtime = datetime.now()
    diff_time = finishtime - starttime
    salida['time']= diff_time.total_seconds()
    salida['changepoint'] = find_changepoints(lista_datos)
    salida['winner'] = winner
    salida['trend']= trendline(lista_datos)
    salida_temp= {}
    salida_temp['status'] = salida
    salida_temp['current'] = 100
    salida_temp['total']=5
    salida_temp['finish'] =5
    salida_temp['result'] ='Task completed'


    # insert json output to mongodb
    import pymongo
    from pymongo import MongoClient
    import os
    import pandas as pd
    import numpy as np

    timecop_backend = os.getenv('mongodb_backend' )
    if timecop_backend != None:
        client = MongoClient(timecop_backend)
        # database
        mongo_db = client["ts"]
        timecop_db= mongo_db["timecop"]
        # data_dict = resultado.to_dict("records")
        lista_puntos = np.arange(0, len(lista_datos),1)
        df = pd.DataFrame(list(zip(lista_puntos, lista_datos)), columns = ['step','value'])
        timecop_db.insert_one({"name":name,"data":salida_temp, "ts": df.to_dict(orient='record')})

    return  salida_temp





@app.route('/multivariate', methods=['POST'])
def multivariate_engine():
    if not request.json:
        abort(400)


    timedata = request.get_json()
    items = timedata['timeseries']
    name = timedata.get('name', 'NA')
    train = timedata.get('train', True)
    restart = timedata.get('restart', False)

    list_var=[]
    for item in items:
        data = item['data']
        if(name != 'NA'):
            sub_name = item['name']

            filename= './lst/'+name + '_' + sub_name +'.lst'
            try:
                with open(filename, 'r') as filehandle:
                    previousList = json.load(filehandle)
            except Exception:
                previousList=[]

            lista = previousList + data
            with open(filename, 'w') as filehandle:
                json.dump(lista,filehandle)


        list_var.append(data)



    lista = timedata['main']
    if(name != 'NA'):
        filename= './lst/'+name+'.lst'
        try:
            with open(filename, 'r') as filehandle:
                previousList = json.load(filehandle)
        except Exception:
            previousList=[]

        lista = previousList + lista
        with open(filename, 'w') as filehandle:
            json.dump(lista,filehandle)

    list_var.append(lista)

    num_fut = int(timedata.get('num_future', 5))
    desv_mae = int(timedata.get('desv_metric', 2))


    desv_mse = 0

    salida = ft.model_multivariate(list_var,num_fut,desv_mae)
    #print(salida)
    return jsonify(salida), 201



@app.route('/back_multivariate', methods=['POST'])

def  back_multivariate_engine():
    if not request.json:
        abort(400)


    timedata = request.get_json()
    items = timedata['timeseries']
    name = timedata.get('name', 'NA')
    train = timedata.get('train', True)

    list_var=[]
    for item in items:
        data = item['data']
        try:

            if(name != 'NA'):
                sub_name = item['name']

                filename= './lst/'+name + '_' + sub_name +'.lst'
                with open(filename, 'r') as filehandle:
                    previousList = json.load(filehandle)

                lista = previousList + data
                with open(filename, 'w') as filehandle:
                    json.dump(lista,filehandle)
        except Exception:
            previousList=[]

        list_var.append(data)



    lista = timedata['main']
    if(name != 'NA'):

        try:
            filename= './lst/'+name+'.lst'
            with open(filename, 'r') as filehandle:
                previousList = json.load(filehandle)
            lista = previousList + lista
            with open(filename, 'w') as filehandle:
                json.dump(lista,filehandle)

        except Exception:
            previousList=[]


    list_var.append(lista)

    num_fut = int(timedata.get('num_future', 5))
    desv_mae = int(timedata.get('desv_metric', 2))


    desv_mse = 0

    #salida = ft.model_multivariate(list_var,num_fut,desv_mae)
    #print(salida)
    #return jsonify(salida), 201

    print ("invoco el backend")
    salida = back_model_multivariate.s(list_var=list_var,num_fut=num_fut,desv_mse=desv_mae,train=train,name=name).apply_async()

    print (salida.id)

        #task = long_task.apply_async()
    valor = {'task_id': salida.id}
    return jsonify(valor), 200
    #return jsonify(salida), 201




@app.route('/back_multivariate_status/<task_id>')
def multivariate_taskstatus(task_id):
    task = back_model_multivariate.AsyncResult(task_id)
    print ("llega aqui")
    print (task)


    if task.state == 'PENDING':
        response = {
            'state': 'Pending',
            'current': 0,
            'total': 1,
            'status': 'Pending...',
            'result': 'Pending'
        }
    if task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            'result': task.info.get('result', ''),
            'response': task.info
        }
    if task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'current': 6,
            'total': 6,
            'result': task.info.get('result', ''),
            'status': task.info.get('status', 'Sucessfully'),
            'task_dump': str(task)
        }
        # if 'result' in task.info:
        #     print ("el result aparece en el SUCCESS")
        #     response['result'] = task.info['result']
        # else:
        #     print ("el result NO aparece en el SUCCESS")


    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            'result': task.info.get('result', ''),
            'response': task.info
        }
    print (task.state)
    print(task.info)
    return jsonify(response)




@celery.task(bind=True)
def back_model_multivariate(self, list_var,num_fut,desv_mse,train=True,name='Test'):

    engines_output={}
    debug = {}
    temp_info = {}

    self.update_state(state='PROGRESS',
        meta={'running': 'LSTM',
            'status': temp_info,
            'total': 2,
            'finish': 2})

    try:
        engines_output['LSTM'] = anomaly_LSTM(list_var,num_fut,desv_mse)
        debug['LSTM'] = engines_output['LSTM']['debug']
        temp_info['LSTM']=engines_output['LSTM']
        self.update_state(state='PROGRESS',
            meta={'running': 'VAR',
                'status': temp_info,
                'total': 2,
                'finish': 1})

        print (engines_output['LSTM'])
    except   Exception as e:
        print(e)
        print ('ERROR: exception executing LSTM')


    try:
        engines_output['VAR'] = anomaly_VAR(list_var,num_fut)
        debug['VAR'] = engines_output['VAR']['debug']
        temp_info['VAR']=engines_output['VAR']
        self.update_state(state='PROGRESS',
            meta={'running': 'VAR',
                'status': temp_info,
                'total': 2,
                'finish': 2})
        print (engines_output['VAR'])
    except   Exception as e:
        print(Exception)
        print("type error: " + str(e))
        print ('ERROR: exception executing VAR')

    try:
        engines_output['VECM'] = anomaly_vecm(list_var,num_fut,desv_mse)
        debug['VECM'] = engines_output['VECM']['debug']
        temp_info['VECM']=engines_output['VECM']
        self.update_state(state='PROGRESS',
            meta={'running': 'VECM',
                'status': temp_info,
                'total': 2,
                'finish': 1})

        print (engines_output['VECM'])
    except   Exception as e:
        print(e)
        print ('ERROR: exception executing VECM')

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
    #return merge_two_dicts(engines_output[winner] , temp)
    salida = merge_two_dicts(engines_output[winner], temp_info)
    salida['winner'] = winner
    salida['trend']= trendline(list_var[0])
    salida_temp= {}
    salida_temp['status'] = salida
    salida_temp['current'] = 100
    salida_temp['total']=4
    salida_temp['finish'] =4
    salida_temp['result'] ='Task completed'

    return  salida_temp




@app.route('/monitoring')
def monitoring():
    model_name = request.args.get('model_name', default = '%', type = str)
    data_models= db.get_all_models(model_name)
    return jsonify(data_models.to_dict(orient='record')),201


@app.route('/monitoring_winners')
def monitoring_winners():
    model_name = request.args.get('model_name', default = '%', type = str)
    data_models= db.get_winners(model_name)
    return jsonify(data_models.to_dict(orient='record')),201


@app.route('/result_list', methods=['POST','GET'])
def result_list():
    from bson import json_util
    timedata = request.get_json()
    collection_ts = timedata.get('collection', 'NA')
    database = timedata.get('database', 'NA')
    timecop_backend = os.getenv('mongodb_backend' )

    if timecop_backend != None:
        url = timecop_backend
    else:
        url = timedata.get('url', 'NA')
    ###"mongodb://username:pwd@ds261570.mlab.com:61570/ts?retryWrites=false"

    import pandas as pd
    import pymongo
    from pymongo import MongoClient
    # Making a Connection with MongoClient
    client = MongoClient(url)
    # database
    db = client[database]
    # collection
    collection_data= db[collection_ts]
    import time
    import json
    from bson import json_util, ObjectId

    #Dump loaded BSON to valid JSON string and reload it as dict
    page_sanitized = json.loads(json_util.dumps(collection_data.find({},{'name':1})))
    return jsonify(page_sanitized), 201

@app.route('/result_document', methods=['POST','GET'])
def result_document():
        from bson import json_util
        timedata = request.get_json()
        database = timedata.get('database', 'NA')
        # url = timedata.get('url', 'NA')
        input_name = timedata.get('name','NA')
        collection_ts = timedata.get('collection_ts','ts')
        collection_timecop = timedata.get('collection_timecop','timecop')
        ###"mongodb://username:pwd@ds261570.mlab.com:61570/ts?retryWrites=false"

        timecop_backend = os.getenv('mongodb_backend' )

        if timecop_backend != None:
            url = timecop_backend
        else:
            url = timedata.get('url', 'NA')
        import pymongo
        from pymongo import MongoClient
        # Making a Connection with MongoClient
        client = MongoClient(url)
        # database
        db = client[database]
        # collection
        data_collection_ts= db[collection_ts]
        ts_data = data_collection_ts.find_one({"name": input_name})
        data_collection_timecop= db[collection_timecop]
        timecop_data = data_collection_timecop.find_one({"name": input_name})
        timecop_data['ts']=ts_data['data']
        import time
        import json
        from bson import json_util, ObjectId

        #Dump loaded BSON to valid JSON string and reload it as dict
        page_sanitized = json.loads(json_util.dumps(timecop_data))
        return jsonify(page_sanitized), 201


@app.route('/')
def index():
    return "Timecop ready to play"

if __name__ == '__main__':
    db.init_database()
    app.run(host = '0.0.0.0',port=PORT)
