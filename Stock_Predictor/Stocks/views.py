from django.shortcuts import render
from .forms import TickerForm
from .forms import EpochForm
from .forms import CryptoForm
from .predictions import Prediction
from .RFClassifier import RFClassifier
from .Crypto import Crypto
from .RFRegressor import RFRegressor
from .SVM import SVM
import gc
import json
import os
import pandas as pd
from django.http import HttpResponse
import random
import datetime
import time


def home(request):
    ticker = TickerForm()
    epoch = EpochForm()
    history_chart = ''
    if request.method == "POST":
        ticker = TickerForm(request.POST)
        if ticker.is_valid():

            # Get Stock Data and Make a History Chart
            tick_obj = TickerForm()
            tick_obj.ticker = ticker.cleaned_data['ticker']
            print(tick_obj.ticker)
            pred = Prediction()
            stock_data = pred.get_data(tick_obj.ticker)
            history_chart = pred.get_history_candlestick(stock_data)
            ticker_str = tick_obj.ticker
            request.session['ticker_str'] = ticker_str

    else:
        ticker = TickerForm()
    print(ticker)
    print(epoch)
    return render(request, 'Stocks/home.html', context={'form': ticker, 'epoch': epoch, 'history_chart': history_chart})


def history(request):
    ticker = ''
    epoch = ''
    if 'ticker_str' in request.session:
        pred = Prediction()
        ticker_str = request.session.get('ticker_str', 'Ticker')
        stock_data = pred.get_data(ticker_str)
        history_chart = pred.get_history_candlestick(stock_data)

    ticker = TickerForm()
    epoch = EpochForm()
    history_chart = ''
    if request.method == "POST":
        ticker = TickerForm(request.POST)
        if ticker.is_valid():

            # Get Stock Data and Make a History Chart
            tick_obj = TickerForm()
            tick_obj.ticker = ticker.cleaned_data['ticker']
            print(tick_obj.ticker)
            pred = Prediction()
            stock_data = pred.get_data(tick_obj.ticker)
            history_chart = pred.get_history_candlestick(stock_data)
            ticker_str = tick_obj.ticker
            request.session['ticker_str'] = ticker_str

    else:
        ticker = TickerForm()
    print(ticker)
    print(epoch)
    return render(request, 'Stocks/history.html', context={'form': ticker, 'epoch': epoch, 'history_chart': history_chart})


def about(request):
    return render(request, 'Stocks/about.html', {'title': 'About'})


def recurrent(request):
    if 'ticker_str' in request.session:
        epochs = 30
        ticker_str = request.session.get('ticker_str', 'Ticker')
        pred = Prediction()
        stock_data = pred.get_data(ticker_str)
        cap = pred.get_market_cap(ticker_str)
        print(cap)
        x_train, x_test, scaler = pred.make_train_test(stock_data)
        x_t, y_t, x_val, y_val, x_test_t, y_test_t = pred.make_test_and_val(
            x_train, x_test)

        model, stock_history = pred.train_model(
            'adam', x_t, y_t, x_val, y_val, epochs)
        model_loss_chart = pred.make_model_loss_chart(
            stock_history, ticker_str)

        prediction, real_data = pred.unscale_data(
            model, x_test_t, y_test_t, scaler)

        prediction_chart = pred.make_prediction_chart(
            prediction, real_data, ticker_str)

        tom_price = 0
        tom_price = pred.predict_tomorrows_price(
            stock_data, model, scaler, y_test_t)

    return render(request, 'Stocks/recurrent.html', context={'title': 'Recurrent',
                                                             'epochs': epochs, 'stock_name': ticker_str, 'tom_price': tom_price,
                                                             'model_loss_chart': model_loss_chart, 'prediction_chart': prediction_chart})


def randomForest(request):
    if 'ticker_str' in request.session:
        ticker_str = request.session.get('ticker_str', 'Ticker')
        rf = RFClassifier()
        rf_data = rf.get_data(ticker_str)
        X, y = rf.make_features(rf_data)
        X_train, X_test, y_train, y_test = rf.split_data(X, y, rf_data)
        model = rf.make_model(X_train, X_test, y_train, y_test)
        rf.make_charts(model, rf_data, X, ticker_str)
        tom_price = rf.predict_tomorrow(model, X)

    return render(request, 'Stocks/random-forests.html', context={'title': 'Random Forest Classifier',
                                                                  'stock_name': ticker_str, 'tom_price': tom_price})


def randomForestRegressor(request):
    if 'ticker_str' in request.session:
        ticker_str = request.session.get('ticker_str', 'Ticker')
        rfr = RFRegressor()
        rfr_data = rfr.get_data(ticker_str)
        model, num_trees, predictions, actual_val, plot, comp_plot = rfr.make_model(
            rfr_data)
        importance_factors, params = rfr.get_importance_factors(model)
        importance = zip(params, importance_factors)

    return render(request, 'Stocks/random-regressor.html', context={'title': 'Random Forest Regressor', 'stock_name': ticker_str,
                                                                    'num_trees': num_trees, 'predictions': predictions,
                                                                    'actual': actual_val, 'plot': plot, 'comp_plot': comp_plot,
                                                                    'importance': importance})


def svm(request):
    if 'ticker_str' in request.session:
        ticker_str = request.session.get('ticker_str', 'Ticker')
        svr = SVM()
        svm_data = svr.get_data(ticker_str)
        x_train_svm, x_test_svm, y_train_svm, y_test_svm = svr.split_data(
            svm_data)
        svr_model, svr_confidence = svr.train_svm_model(
            x_train_svm, x_test_svm, y_train_svm, y_test_svm)
        lr_model, lr_confidence = svr.train_lr_model(
            x_train_svm, x_test_svm, y_train_svm, y_test_svm)

        svm_predictions, actual_values = svr.validate(svr_model, svm_data)
        lr_predictions, actual_values = svr.validate(lr_model, svm_data)

        svm_plot = svr.make_chart(
            lr_predictions, svm_predictions, actual_values)

    return render(request, 'Stocks/svm.html', context={'title': 'SVM', 'stock_name': ticker_str, 'plot': svm_plot,
                                                       'lr_confidence': round(lr_confidence, 2), 'svr_confidence': round(svr_confidence, 2)})


def crypto(request):

    gc.collect()

    crypto = CryptoForm()
    crypto_history_chart = ''
    if request.method == "POST":
        crypto = CryptoForm(request.POST)
        if crypto.is_valid():

            crypt_obj = CryptoForm()
            crypt_obj.crypto = crypto.cleaned_data['crypto']
            coin = Crypto(crypt_obj.crypto)
            data = coin.get_crypto_data()
            crypto_history_chart = coin.get_crypto_candlestick(data)
            crypto_str = crypt_obj.crypto
            request.session['crypto_str'] = crypto_str
        else:
            crypto = CryptoForm()

    return render(request, 'Stocks/crypto.html', context={'title': 'crypto', 'form': crypto,
                                                          'crypto_history_chart': crypto_history_chart})


def cryptoModel(request):
    if 'crypto_str' in request.session:
        crypto_str = request.session.get('crypto_str', 'Crypto')
        print(crypto_str)
        coin = Crypto(crypto_str)
        data = coin.get_crypto_data()
        tom_price, model_loss_chart, prediction_chart = coin.make_model(data)

    return render(request, 'Stocks/crypto-model.html', context={'title': 'Crypto Model',
                                                                'crypto_name': crypto_str, 'tom_price': tom_price,
                                                                'model_loss_chart': model_loss_chart, 'prediction_chart': prediction_chart})
