from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import statsmodels.api as sm

def trendline(data, order=1):

    coeffs = np.polyfit(np.arange(0,len(data)), list(data), order)
    slope = coeffs[-2]
    return float(slope)




def reshape_array(x):
    assert len(x.shape) == 2, 'input np.array should be in the format: samples, timesteps'
    if len(x.shape) == 2:
        nb_samples, nb_timestamps = x.shape
        return x.reshape((nb_samples, nb_timestamps, 1))


def seasonal_options (a):
  print(" Starting seasonal finding")
  print(a)
  x =sm.tsa.stattools.pacf(a)

  possible =[]
  for i in range(4, len(x)-6):
    before2 = x[i-2]
    before= x[i-1]
    period = x[i]
    last = x[i+1]
    last2 = x[i+2]
    if (before2 < before < period > last ):
      possible.append(i-1)
  print ("Finishing seasonal finding")
  return possible

def windows(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))


# Define the model.
def fit_model_new(train_X, train_Y, window_size = 1):
    model2 = Sequential()
    model2.add(LSTM(input_shape = (window_size, 1),
               units = window_size,
               return_sequences = True))
    model2.add(Dropout(0.5))
    model2.add(LSTM(256))
    model2.add(Dropout(0.5))
    model2.add(Dense(1))
    model2.add(Activation("linear"))
    model2.compile(loss = "mse",
              optimizer = "adam")
    model2.summary()

    # Fit the first model.
    model2.fit(train_X, train_Y, epochs = 80,
              batch_size = 1,
              verbose = 2)
    return(model2)


def predict_and_score(model, X, Y,scaler):
    # Make predictions on the original scale of the data.
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = mean_squared_error(orig_data[0], pred[:, 0])
    mae = mean_absolute_error(orig_data[0], pred[:, 0])
    return(score, pred, pred_scaled,mae)



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def create_train_test(lista_puntos, lista_datos):
    df = pd.DataFrame()
    df['puntos'] = lista_puntos
    df['valores'] = lista_datos

    df.set_index('puntos',inplace=True,drop=False)
    tam_train = int(len(df)*0.7)
    print (" train length" + str(tam_train))

    df_train = df[:tam_train]
    df_test = df[tam_train:]
    return df, df_train, df_test
