import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to load data
def load_data(ticker, start, end):
    df = yf.download(ticker, start, end)
    df.dropna(inplace=True)

    # Moving Averages
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    # Bollinger Bands
    df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Returns
    df['returns'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)

    df = df.drop('Adj Close', axis=1)

    return df

# Function to prepare data for the model
def prepare_data(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA200', 'SMA50', 'RSI', 'MACD', 'Signal_Line', 'BB_upper', 'BB_lower', 'returns']
    X = df[features]
    y = df['Close']

    X = X.dropna()
    y = y[X.index]

    scaler = MinMaxScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=10, shuffle=False)

    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, scaler

# Function to create and train the model
def create_model(X_train):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Streamlit app
st.title("Stock Price Prediction")

ticker = st.text_input("Enter Stock Ticker", "AMZN")
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

if ticker:
    df = load_data(ticker, START, TODAY)

    st.subheader(f"Recent Data for {ticker}")
    st.write(df.tail())

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    model = create_model(X_train)

if st.button("Train Model"):
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    st.success("Model trained!")

    # Predict on test data
    predictions = model.predict(X_test)
    
    st.subheader("Predictions vs Actual")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, predictions, label='Predicted')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)

    # Predict the next day
    next_day_prediction = model.predict(X_test[-1].reshape(1, 1, X_test.shape[2]))
    next_day_price = scaler.inverse_transform(np.hstack((np.zeros((1, X_test.shape[2]-1)), next_day_prediction.reshape(-1, 1))))[:, -1]
    
    st.subheader("Next Day Prediction")
    st.write(f"Predicted Price for Next Day: {next_day_price[0]:.2f}")


