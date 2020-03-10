from flask import request, Flask, jsonify, abort
from flask_cors import CORS
import json

import engines.functions_timeseries as ft
import engines.BBDD as db
import os
from celery import Celery



# import engines functions_timeseries
from engines.helpers import merge_two_dicts
from engines.var import anomaly_VAR, univariate_anomaly_VAR,univariate_forecast_VAR
from engines.holtwinter import anomaly_holt,forecast_holt
from engines.auto_arima import anomaly_AutoArima
from engines.lstm import anomaly_LSTM, anomaly_uni_LSTM

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

    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    if task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': task.info.get('status', 'Running...')
        }
    if task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'current': 4,
            'total': 4,
            'result': task.info.get('result', ''),
            'status': task.info.get('status', 'Sucessfully'),
            'task_dump': str(task)
        }
        if 'result' in task.info:
            print ("el result aparece en el SUCCESS")
            response['result'] = task.info['result']
        else:
            print ("el result NO aparece en el SUCCESS")


    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            'result': task.info.get('result', ''),
            'response': task.info
        }
    else:

        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
            'result': task.info
        }
    print (task.state)
    print(task.info)
    return jsonify(response)




############################backen functions


@celery.task(bind=True)
def back_model_univariate(self, lista_datos,num_fut,desv_mse,train,name):
    engines_output={}
    debug = {}

    temp_info = {}

    self.update_state(state='PROGRESS',
                      meta={'running': 'LSTM',
                            'status': '',
                            'total': 4,
                            'finish': 0 })
    if not train:

        (model_name,model,params)=db.get_best_model('winner_'+name)
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
            temp_info['LSTM']=engines_output['LSTM']
            self.update_state(state='PROGRESS',
                      meta={'running': 'anomaly_AutoArima',
                            'status': temp_info,
                            'total': 4,
                            'finish': 1})
        except Exception as e:
            print(e)
            print ('ERROR: exception executing LSTM univariate')


        try:
            if (len(lista_datos) > 100):
                #new_length=
                lista_datos_ari=lista_datos[len(lista_datos)-100:]
            engines_output['arima'] = anomaly_AutoArima(lista_datos_ari,num_fut,len(lista_datos),desv_mse)
            debug['arima'] = engines_output['arima']['debug']
            temp_info['arima']=engines_output['arima']
            self.update_state(state='PROGRESS',
                      meta={'running': 'VAR',
                            'status': temp_info,
                            'total': 4,
                            'finish': 2})
        except  Exception as e:
            print(e)
            print ('ERROR: exception executing Autoarima')

        try:
            if (train):
                engines_output['VAR'] = univariate_anomaly_VAR(lista_datos,num_fut,name)
                debug['VAR'] = engines_output['VAR']['debug']
            else:
                engines_output['VAR'] = univariate_forecast_VAR(lista_datos,num_fut,name)
                debug['VAR'] = engines_output['VAR']['debug']
            temp_info['VAR'] = engines_output['VAR']
            self.update_state(state='PROGRESS',
                      meta={'running': 'Holtwinters',
                            'status': temp_info,
                            'total': 4,
                            'finish': 3})

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

            temp_info['Holtwinters'] = engines_output['Holtwinters']
            self.update_state(state='PROGRESS',
                      meta={'running': 'Holtwinters',
                            'status': temp_info,
                            'total': 4,
                            'finish': 4})

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

        db.new_model('winner_'+name, winner, pack('N', 365),'',0)


        print (winner)

    print ("el ganador es " + str(winner))
    print (engines_output[winner])
    temp= {}
    temp['debug']=debug

#    return merge_two_dicts(engines_output[winner] , temp)
    salida = merge_two_dicts(engines_output[winner], temp_info)
    salida['winner'] = winner
    salida_temp= {}
    salida_temp['status'] = salida
    salida_temp['current'] = 100
    salida_temp['total']=100
    salida_temp['result'] ='Task completed'

    return  salida_temp





@app.route('/multivariate', methods=['POST'])
def multivariate_engine():
    if not request.json:
        abort(400)


    timedata = request.get_json()
    items = timedata['timeseries']
    name = timedata.get('name', 'NA')
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


@app.route('/')
def index():
    return "Timecop ready to play"

if __name__ == '__main__':
    db.init_database()
    app.run(host = '0.0.0.0',port=PORT)
