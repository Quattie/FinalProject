import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from plotly.offline import plot
import plotly.graph_objs as go

forecast_out = 30


class SVM():

    def get_data(self, user_ticker):
        stock = yf.Ticker(user_ticker)
        df = stock.history(period='max')
        df = df[['Close']]
        df.head()
        return df

    def split_data(self, df):
        df['Prediction'] = df[['Close']].shift(-forecast_out)
        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        return x_train, y_train, x_test, y_test

    def train_svm_model(self, x_train, y_train, x_test, y_test):
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_rbf.fit(x_train, y_train)
        svm_confidence = svr_rbf.score(x_test, y_test)
        print('SVM Confidence:', svm_confidence)
        return svr_rbf, svm_confidence

    def train_lr_model(self, x_train, y_train, x_test, y_test):
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        lr_confidence = lr.score(x_test, y_test)
        print('LR Confidence: ', lr_confidence)
        return lr, lr_confidence

    def validate(self, model, df):
        actual_values = df['Close']
        actual_values = actual_values[-forecast_out:]
        x_validation = np.array(df.drop(['Prediction'], 1))[-forecast_out*2:]
        # Shave off the last 30 days for validation to compare
        x_validation = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        model_prediction = model.predict(x_validation)
        return model_prediction, actual_values

    def make_chart(self, lr_predictions, svm_predictions, actual_values):
        trace1 = go.Scatter(
            y=svm_predictions,
            mode='lines',
            name='SVM'
        )
        trace2 = go.Scatter(
            y=actual_values,
            mode='lines',
            name='Actual Prices'
        )
        trace3 = go.Scatter(
            y=lr_predictions,
            mode='lines',
            name='Actual Prices'
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
        figure = go.Figure(data=plot_data, layout=layout)
        figure.layout.template = 'plotly_dark'
        plot_div = plot(figure, output_type='div', include_plotlyjs=False)
        return plot_div
