from flask import request, Flask, jsonify, abort
from flask_cors import CORS
import json

import engines.functions_timeseries as ft
import engines.BBDD as db
import os


app = Flask(__name__)
CORS(app)


app.config.from_pyfile(os.path.join(".", "config/app.cfg"), silent=False)

db.init_database()

DB_NAME= app.config.get("DB_NAME")
PORT = app.config.get("PORT")

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


@app.route('/')
def index():
    return "Timecop ready to play"

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=PORT)
