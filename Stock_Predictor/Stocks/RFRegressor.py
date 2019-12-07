from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib.pylab import rcParams
import yfinance as yf
import pandas as pd
from datetime import date
import quandl
import math
import numpy as np
import os
from plotly.offline import plot
import plotly.graph_objs as go

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
rcParams['figure.figsize'] = 20, 10

# importing required libraries


def rmse(x, y): return math.sqrt(((x-y)**2).mean())


def print_score(m, X_train, y_train, X_val, y_val):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_val), y_val),
           m.score(X_train, y_train), m.score(X_val, y_val)]
    print(
        f"rmse train {res[0]}, rmse val {res[1]}, r^2 train {res[2]}, r^2 val {res[3]}")


class RFRegressor():

    def get_data(self, user_ticker):

        tick = yf.Ticker(user_ticker)
        stock = tick.history(period='max')
        today = date.today()
        formed_date = today.strftime("%Y-%m-%d")
        stock = stock.loc['2002-01-01':formed_date]
        stock.tail()

        # Make features
        stock['Change'] = stock['Open'] - stock['Close']
        stock['Intraday_Change'] = stock['High'] - stock['Low']
        stock['Movement'] = stock['High'] - stock['Low'] / stock['High']
        stock['Day_of_Week'] = stock.index.dayofweek
        stock['Day_of_Year'] = stock.index.dayofyear
        stock['y'] = stock['Close']
        stock['Date'] = stock.index
        stock['Year'] = stock['Date'].dt.year
        stock['Month'] = stock['Date'].dt.month
        stock.fillna(0)
        print(stock.isnull().sum())
        return stock

    def make_model(self, stock):
        TOTAL = stock.count()[0]
        N_VALID = 120  # Three months
        TRAIN = TOTAL - N_VALID
        params = ['Open', 'Close', 'High', 'Low', 'Volume', 'Change',
                  'Intraday_Change', 'Movement', 'Year', 'Month', 'Day_of_Week', 'Day_of_Year']
        stock_data = stock[params]

        X_df = stock_data
        y_df = stock['y']
        X_train, X_val = X_df[:TRAIN], X_df[TRAIN:]
        y_train, y_val = y_df[:TRAIN], y_df[TRAIN:]

        estimators_num = 200
        model = RandomForestRegressor(
            n_estimators=estimators_num, bootstrap=True, min_samples_leaf=25)
        model.fit(X_train, y_train)
        estimator = model.estimators_[0]
        print_score(model, X_train, y_train, X_val, y_val)
        print("Mean of y train ", y_train.mean())

        model.predict(X_val)[0]
        x_val_predicts = model.predict(X_val)
        close = stock_data['Close'].values

        preds = np.stack([t.predict(X_val) for t in model.estimators_])
        num_trees = len(preds[:, 0])
        predictions = round(x_val_predicts[-1], 2)
        actual_val = round(close[-1], 2)

        print(f"Trees: {num_trees},",
              f"Mean of 0th row for prediction from all trees: : {np.mean(preds[:,0])},", f"Actual y: {y_val[0]}")

        plot_data = [metrics.r2_score(y_val, np.mean(
            preds[:i+1], axis=0)) for i in range(estimators_num)]
        trace1 = go.Scatter(
            y=plot_data,
            mode='lines',
            name='Model Accuracy'
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

        comp_trace1 = go.Scatter(
            y=x_val_predicts,
            mode='lines',
            name='Predictions'
        )
        comp_trace2 = go.Scatter(
            y=close[-120:],
            mode='lines',
            name='Actual'
        )
        layout2 = go.Layout(
            autosize=True,
            xaxis=dict(
                autorange=True
            ),
            yaxis=dict(
                autorange=True
            )
        )
        comp_plot_data = [comp_trace1, comp_trace2]
        figure2 = go.Figure(data=comp_plot_data, layout=layout2)
        figure2.layout.template = 'plotly_dark'
        comp_plot_div = plot(figure2, output_type='div',
                             include_plotlyjs=False)

        return model, num_trees, predictions, actual_val, plot_div, comp_plot_div

    def get_importance_factors(self, model):
        params = ['Open', 'Close', 'High', 'Low', 'Volume', 'Change',
                  'Intraday_Change', 'Movement', 'Year', 'Month', 'Day_of_Week', 'Day_of_Year']
        features = []
        for feature in model.feature_importances_:
            features.append(round(feature, 4))
        return features, params
