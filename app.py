# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

from data_collector import download_data
from strategy import moving_average_crossover, rsi_strategy, bollinger_bands_strategy
from backtester import backtest
from ml_model import train_model, generate_signals
from sentiment_analysis import integrate_sentiment

st.title("Algorithmic Trading Bot")

# Strategy Descriptions
strategy_descriptions = {
    "Moving Average Crossover": "A strategy that uses short and long-term moving averages to generate buy/sell signals.",
    "RSI Strategy": "A strategy that uses the Relative Strength Index (RSI) to identify overbought or oversold conditions.",
    "Bollinger Bands Strategy": "A strategy that uses Bollinger Bands to identify price breakouts and volatility.",
    "ML Model": "A strategy that uses a machine learning model to predict future price movements.",
}

ticker = st.text_input("Ticker Symbol", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

strategy_option = st.selectbox(
    "Select a Strategy",
    ("Moving Average Crossover", "RSI Strategy", "Bollinger Bands Strategy", "ML Model"),
    format_func=lambda x: f"{x}: {strategy_descriptions[x]}"
)

stop_loss = st.slider("Stop-Loss Percentage", min_value=0.01, max_value=0.2, step=0.01, value=0.05)
take_profit = st.slider("Take-Profit Percentage", min_value=0.01, max_value=0.2, step=0.01, value=0.1)

if st.button("Run Strategy"):
    data = download_data(ticker, start_date, end_date)
    st.write("### Historical Data", data.tail())
    
    if strategy_option == "Moving Average Crossover":
        short_window = st.number_input("Short Moving Average Window", min_value=1, value=50)
        long_window = st.number_input("Long Moving Average Window", min_value=1, value=200)
        strategy_data = moving_average_crossover(data, short_window, long_window)
    elif strategy_option == "RSI Strategy":
        window = st.number_input("RSI Window", min_value=1, value=14)
        overbought = st.number_input("RSI Overbought Level", min_value=1, max_value=100, value=70)
        oversold = st.number_input("RSI Oversold Level", min_value=1, max_value=100, value=30)
        strategy_data = rsi_strategy(data, window, overbought, oversold)
    elif strategy_option == "Bollinger Bands Strategy":
        window = st.number_input("Bollinger Bands Window", min_value=1, value=20)
        num_std = st.number_input("Number of Standard Deviations", min_value=0.1, max_value=5.0, step=0.1, value=2.0)
        strategy_data = bollinger_bands_strategy(data, window, num_std)
    elif strategy_option == "ML Model":
        model, scaler, accuracy = train_model(data)
        st.write(f"### ML Model Accuracy: {accuracy:.2f}")
        strategy_data = generate_signals(data, model, scaler)
    
    st.write("### Strategy Data", strategy_data.tail())
    
    # Integrate sentiment analysis
    strategy_data = integrate_sentiment(strategy_data, ticker)
    st.write("### Sentiment Score", strategy_data['Sentiment'].iloc[-1])
    
    backtest_data, total_return, sharpe_ratio, max_drawdown = backtest(strategy_data, stop_loss=stop_loss, take_profit=take_profit)
    st.write("### Backtest Data", backtest_data.tail())
    
    st.write("### Performance Metrics")
    st.write(f"**Total Return**: ${total_return:,.2f}")
    st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")
    st.write(f"**Max Drawdown**: {max_drawdown:.2%}")

    fig = px.line(backtest_data, x=backtest_data.index, y=['Close', 'Balance'],
                  labels={'value': 'Price'}, title="Strategy and Balance Over Time")
    st.plotly_chart(fig)
    
    if strategy_option == "Moving Average Crossover":
        fig_ma = px.line(backtest_data, x=backtest_data.index, y=['Short_MA', 'Long_MA'],
                         labels={'value': 'Moving Average'}, title="Moving Averages Over Time")
        st.plotly_chart(fig_ma)
    
    if strategy_option == "RSI Strategy":
        fig_rsi = px.line(backtest_data, x=backtest_data.index, y=['RSI'],
                          labels={'value': 'RSI'}, title="RSI Over Time")
        st.plotly_chart(fig_rsi)
    
    if strategy_option == "Bollinger Bands Strategy":
        fig_bb = px.line(backtest_data, x=backtest_data.index, y=['Rolling_Mean', 'Bollinger_Upper', 'Bollinger_Lower'],
                         labels={'value': 'Bollinger Bands'}, title="Bollinger Bands Over Time")
        st.plotly_chart(fig_bb)
