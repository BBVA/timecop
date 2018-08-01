from flask import request

from flask import Flask, jsonify

from flask_cors import CORS


import numpy as np
import pandas as pd
from engines.functions_timeseries import *


from flask import abort
app = Flask(__name__)
CORS(app)


@app.route('/univariate/get', methods=['POST'])
def univariate_engine():
    if not request.json:
        abort(400)
        
        
    timedata = request.get_json()
    lista=timedata['data']
    
    num_fut = 5
    desv_mse = 0
    


    #dict_Anomaly_AutoArima,dict_last_5_AutoArima,dict_fore_AutoArima,dict_Anomaly_holt,dict_last_5_holt,dict_fore_holt = model_univariate(lista,num_fut,desv_mse)
    salida = model_univariate(lista,num_fut,desv_mse)
    return jsonify(salida), 201


@app.route('/multivariate/get', methods=['POST'])
def multivariate_engine():
    if not request.json:
        abort(400)
        
        
    timedata = request.get_json()
    items = timedata['timeseries']
    list_var=[]
    for item in items:
        #print (data)
        data = item['data']
        list_var.append(data)
        
    list_var.append(timedata['main'])
    
    num_fut = 5
    desv_mse = 0
    


    #dict_Anomaly_AutoArima,dict_last_5_AutoArima,dict_fore_AutoArima,dict_Anomaly_holt,dict_last_5_holt,dict_fore_holt = model_univariate(lista,num_fut,desv_mse)
    salida = model_multivariate(list_var,num_fut,desv_mse)
    print(salida)
    #return json, 201
    return jsonify(salida), 201


@app.route('/')
def index():
    return "Timecop ready to play"

if __name__ == '__main__':
    app.run(debug=True)#,host = '0.0.0.0')
