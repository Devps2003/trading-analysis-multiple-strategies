# ml_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def prepare_features(data):
    data['Return'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].diff().where(data['Close'].diff() > 0, 0).rolling(window=14).mean() / 
                                 -data['Close'].diff().where(data['Close'].diff() < 0, 0).rolling(window=14).mean()))
    data.dropna(inplace=True)
    return data

def train_model(data):
    data = prepare_features(data)
    X = data[['Return', 'MA_5', 'MA_20', 'RSI']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    return model, scaler, accuracy

def generate_signals(data, model, scaler):
    data = prepare_features(data)
    X = data[['Return', 'MA_5', 'MA_20', 'RSI']]
    X_scaled = scaler.transform(X)
    data['Signal'] = model.predict(X_scaled)
    data['Position'] = data['Signal'].diff()
    return data
