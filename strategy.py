# strategy.py

import pandas as pd
import numpy as np

def moving_average_crossover(data, short_window=50, long_window=200):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data

def rsi_strategy(data, window=14, overbought=70, oversold=30):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Signal'] = 0
    data['Signal'] = np.where(data['RSI'] > overbought, -1, data['Signal'])
    data['Signal'] = np.where(data['RSI'] < oversold, 1, data['Signal'])
    data['Position'] = data['Signal'].diff()
    return data

def bollinger_bands_strategy(data, window=20, num_std=2):
    data['Rolling_Mean'] = data['Close'].rolling(window).mean()
    data['Bollinger_Upper'] = data['Rolling_Mean'] + (data['Close'].rolling(window).std() * num_std)
    data['Bollinger_Lower'] = data['Rolling_Mean'] - (data['Close'].rolling(window).std() * num_std)
    data['Signal'] = 0
    data['Signal'] = np.where(data['Close'] > data['Bollinger_Upper'], -1, data['Signal'])
    data['Signal'] = np.where(data['Close'] < data['Bollinger_Lower'], 1, data['Signal'])
    data['Position'] = data['Signal'].diff()
    return data
