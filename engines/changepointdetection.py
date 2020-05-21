import ruptures as rpt
import pandas as  pd
import numpy as np

def find_changepoints(lista_datos):
    #Convert the time series values to a numpy 1D array
    points=np.array(lista_datos)

    #Changepoint detection with the Binary Segmentation search method
    model = "l2"
    algo = rpt.Binseg(model=model).fit(points)
    my_bkps = algo.predict(pen=np.log(len(lista_datos))*6**2)
    output= pd.DataFrame(my_bkps,columns=['step'])
    return(output.to_dict(orient='record'))
