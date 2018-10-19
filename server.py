from flask import request, Flask, jsonify, abort
from flask_cors import CORS
import json

import engines.functions_timeseries as ft

app = Flask(__name__)
CORS(app)


@app.route('/univariate/get', methods=['POST'])
def univariate_engine():
    if not request.json:
        abort(400)
        
        
    timedata = request.get_json()
    lista=timedata['data']
    
    num_fut = int(timedata.get('num_future', 5))
    desv_mae = int(timedata.get('desv_metric', 2))
    name = timedata.get('name', 'NA')
    
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
    

    #desv_mse = 0
    
    salida = ft.model_univariate(lista,num_fut,desv_mae)
    return jsonify(salida), 201


@app.route('/multivariate/get', methods=['POST'])
def multivariate_engine():
    if not request.json:
        abort(400)
        
        
    timedata = request.get_json()
    items = timedata['timeseries']
    name = timedata.get('name', 'NA')
    list_var=[]
    for item in items:
        data = item['data']
        sub_name = item['name']
        if(name != 'NA'):
            
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
    app.run(debug=True)#,host = '0.0.0.0')
