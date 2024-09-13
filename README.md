# stockprediction

Overview
Stock Price Prediction Web App that uses historical stock data from yfinance and technical indicators to predict future stock prices on a daily time period. The app is built with Streamlit an LSTM neural network for predictions using technical indicators such as Moving Averages, Bollinger Bands, RSI, MACD, and percentage returns.

Features
  Stock Data: Fetches historical data from Yahoo Finance for any stock ticker.
  Technical Indicators: Computes SMA, Bollinger Bands, RSI, MACD, and returns.
  LSTM Model: Predicts stock prices using an LSTM model built in TensorFlow.
  Next Day Prediction: Predicts the next day’s stock price.

Libraries
Streamlit, yfinance, Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow

How It Works
  Data Loading: Fetches stock data for a specified ticker.
  Feature Engineering: Computes key technical indicators.
  Data Preprocessing: Normalizes data and splits into training and testing sets.
  LSTM Model: Trains on historical data and predicts stock prices.
  Visualization: Plots predicted vs actual stock prices, plus next day prediction.

Setup
Install required libraries:
pip install -r requirements

Run the app:
streamlit run app.py

Future Enhancements
  Add news sentiment analysis.
  Add more technical indicators.
  Hyperparamter tuning and testing different model structures

Sources referenced - 
https://www.tradingsim.com/resources/stock-trading-indicators
https://www.listendata.com/2021/02/calculate-technical-indicators-for.html#relative_strength_index_rsi
https://medium.com/@financial_python/building-a-macd-indicator-in-python-190b2a4c1777
