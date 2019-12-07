import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.random import seed
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook
import pandas_datareader.data as pdr

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from plotly.offline import plot
import plotly.graph_objs as go

import matplotlib
matplotlib.use('Agg')

TIME_STEPS = 5
BATCH_SIZE = 2


def build_timeseries(mat, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    print("Length of time-series i/o", x.shape, y.shape)
    return x, y


def trim_data(mat, batch_size):
    num_rows_dropped = mat.shape[0] % batch_size
    if(num_rows_dropped > 0):
        return mat[:-num_rows_dropped]
    else:
        return mat


def make_prediction_chart(self, y_pred_org, y_test_org_t):

    pred = []
    for x in y_pred_org:
        pred.append(x[0])
    y_test_org_t.tolist()
    # Numpy Array Reversal
    y_test_org_t = y_test_org_t[::-1]
    # List Reversal
    pred.reverse()
    trace1 = go.Scatter(
        y=pred,
        mode='lines',
        name='Predictions'
    )
    trace2 = go.Scatter(
        y=y_test_org_t,
        mode='lines',
        name='Actual Values'
    )
    layout = go.Layout(
        autosize=True,
        xaxis=dict(
            autorange="reversed"
        ),
        yaxis=dict(
            autorange=True
        )
    )
    plot_data = [trace1, trace2]
    figure = go.Figure(data=plot_data, layout=layout)
    figure.layout.template = 'plotly_dark'
    plot_div = plot(figure, output_type='div', include_plotlyjs=False)
    return plot_div


class Crypto:

    def __init__(self, coin):
        self.coin = coin
        if coin == 'bitcoin':
            self.tag = 'BTC'
        elif coin == 'ethereum':
            self.tag = 'ETH'
        elif coin == 'litecoin':
            self.tag = 'LTC'
        elif coin == 'ltc':
            self.coin = 'litecoin'
            self.tag = 'LTC'
        elif coin == 'eth':
            self.coin = 'ethereum'
            self.tag = 'ETH'
        elif coin == 'btc':
            self.coin = 'bitcoin'
            self.tag = 'BTC'
        else:
            raise ValueError(
                'We only support Bitcoin, Ethereum and Litecoin right now')

    def get_crypto_data(self):

        end = datetime.today()
        start = datetime(end.year-8, end.month, end.day)
        data = pdr.DataReader(self.tag + '-USD', 'yahoo', start, end)
        return data

    def get_crypto_candlestick(self, data):

        data = data.iloc[::-1]
        trace1 = go.Candlestick(
            x=data.index,
            open=data[data.columns[0]],
            high=data[data.columns[1]],
            low=data[data.columns[2]],
            close=data[data.columns[3]]
        )

        layout = go.Layout(
            autosize=True,
            xaxis=dict(
                autorange=True
            ),
            yaxis=dict(
                autorange=True
            )
        )
        plot_data = [trace1]
        figure = go.Figure(data=plot_data, layout=layout)
        figure.layout.template = 'plotly_dark'
        plot_div = plot(figure, output_type='div', include_plotlyjs=False)
        return plot_div

    def make_model(self, data):
        # Make training set 80% of the data
        train_cols = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

        btc_train, btc_test = train_test_split(
            data, train_size=0.8, test_size=0.2, shuffle=False)
        x = btc_train.loc[:, train_cols].values
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x)
        x_test = scaler.fit_transform(data.loc[:, train_cols])

        x_t, y_t = build_timeseries(x_train, 3)
        x_t = trim_data(x_t, BATCH_SIZE)
        y_t = trim_data(y_t, BATCH_SIZE)
        x_temp, y_temp = build_timeseries(x_test, 3)
        x_val, x_test_t = np.split(trim_data(x_temp, BATCH_SIZE), 2)
        y_val, y_test_t = np.split(trim_data(y_temp, BATCH_SIZE), 2)
        print(x_test_t.shape)

        model = Sequential()
        model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0,
                       recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()

        btc_history = model.fit(x_t, y_t, epochs=15, verbose=2,
                                batch_size=BATCH_SIZE, shuffle=False,
                                validation_data=(trim_data(x_val, BATCH_SIZE), trim_data(y_val, BATCH_SIZE)))

        y_pred = model.predict(
            trim_data(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        print(x_test_t.shape)
        y_pred - y_pred.flatten()
        y_test_t = trim_data(y_test_t, BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is ", error, y_pred.shape, y_test_t.shape)

        y_pred_org = (y_pred * scaler.data_range_[3] + scaler.data_min_[3])
        y_test_org_t = (y_test_t * scaler.data_range_[3] + scaler.data_min_[3])

        print(y_pred_org.shape)
        print(y_test_org_t.shape)

        trace1 = go.Scatter(
            y=btc_history.history['val_loss'],
            mode='lines',
            name='Model Accuracy'
        )
        trace2 = go.Scatter(
            y=btc_history.history['loss'],
            mode='lines',
            name='Test Data'
        )
        layout = go.Layout(
            autosize=True,
            xaxis=dict(
                autorange=True
            ),
            yaxis=dict(
                autorange=True
            )
        )
        plot_data = [trace1, trace2]
        model_figure = go.Figure(data=plot_data, layout=layout)
        model_figure.layout.template = 'plotly_dark'
        model_loss_plot_div = plot(
            model_figure, output_type='div', include_plotlyjs=False)

        prediction_chart = make_prediction_chart(
            self, y_pred_org, y_test_org_t)

        next_pred = data[-7:]
        train_cols = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

        x_pred_values = next_pred.loc[:, train_cols].values
        predx = scaler.fit_transform(x_pred_values)
        x_fut, y_fut = build_timeseries(predx, 3)

        y_test_t = trim_data(y_test_t, BATCH_SIZE)

        future_pred = model.predict(
            trim_data(x_fut, BATCH_SIZE), batch_size=BATCH_SIZE)

        future_pred - future_pred.flatten()
        y_fut = trim_data(y_fut, BATCH_SIZE)

        fut_pred_org = (
            future_pred * scaler.data_range_[3] + scaler.data_min_[3])
        y_fut = (y_fut * scaler.data_range_[3] + scaler.data_min_[3])

        print(next_pred)
        print(y_fut)
        print(fut_pred_org.shape)
        print(fut_pred_org[0][-1])
        return float("{0:.2f}".format(fut_pred_org[0][0])), model_loss_plot_div, prediction_chart
