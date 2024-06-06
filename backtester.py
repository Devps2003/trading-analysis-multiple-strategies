# backtester.py

import pandas as pd
import numpy as np

def calculate_performance_metrics(data, initial_balance=10000):
    total_return = data['Balance'].iloc[-1] - initial_balance
    data['Daily_Return'] = data['Balance'].pct_change()
    sharpe_ratio = np.mean(data['Daily_Return']) / np.std(data['Daily_Return']) * np.sqrt(252)
    drawdown = data['Balance'] / data['Balance'].cummax() - 1
    max_drawdown = drawdown.min()
    return total_return, sharpe_ratio, max_drawdown

def backtest(data, initial_balance=10000, share_size=10, stop_loss=0.05, take_profit=0.1):
    balance = initial_balance
    positions = 0
    entry_price = 0
    balance_history = []

    for i in range(len(data)):
        if data['Position'][i] == 1 and balance >= data['Close'][i] * share_size:
            positions += share_size
            balance -= data['Close'][i] * share_size
            entry_price = data['Close'][i]
        elif data['Position'][i] == -1 and positions > 0:
            balance += data['Close'][i] * positions
            positions = 0
        elif positions > 0:
            current_price = data['Close'][i]
            if current_price <= entry_price * (1 - stop_loss) or current_price >= entry_price * (1 + take_profit):
                balance += current_price * positions
                positions = 0
        
        balance_history.append(balance + positions * data['Close'][i])

    data['Balance'] = balance_history
    total_return, sharpe_ratio, max_drawdown = calculate_performance_metrics(data, initial_balance)
    return data, total_return, sharpe_ratio, max_drawdown

if __name__ == "__main__":
    data = pd.read_csv("strategy_output.csv", index_col='Date', parse_dates=True)
    result, total_return, sharpe_ratio, max_drawdown = backtest(data)
    result.to_csv("backtest_output.csv")
