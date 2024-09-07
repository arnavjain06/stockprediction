import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import streamlit as st

st.title('Stock prediction')



def load_data(stock):
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    df = yf.download (stock, START, TODAY)
    return df

user_choice = st.text_input("Enter the stock you wish to predict: ")

data_load_state = st.text('Loading data...')

data = load_data(user_choice)

data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.dataframe(data)

st.subheader(user_choice + " Line graph")
st.line_chart(data['Close'])