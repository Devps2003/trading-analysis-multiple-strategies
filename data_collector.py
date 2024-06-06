# data_collector.py

import yfinance as yf
import pandas as pd

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    data = download_data(ticker, start_date, end_date)
    data.to_csv("data.csv")
