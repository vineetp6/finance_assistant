import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def predict_price(ticker):
    df = get_stock_data(ticker)
    df['Date'] = df.index.map(pd.Timestamp.toordinal)
    X = df[['Date']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_date = (datetime.now() + timedelta(days=30)).toordinal()
    future_price = model.predict([[future_date]])
    
    return future_price[0]

st.title("Financial Virtual Assistant")
st.sidebar.header("User Input")

# User inputs
ticker = st.sidebar.text_input("Stock Ticker", value='AAPL')
investment_goal = st.sidebar.selectbox("Investment Goal", ["Growth", "Income", "Preservation"])
risk_tolerance = st.sidebar.slider("Risk Tolerance", 1, 10, 5)

# Display stock data
st.header(f"Stock Data for {ticker}")
stock_data = get_stock_data(ticker)
st.line_chart(stock_data['Close'])

# Display prediction
predicted_price = predict_price(ticker)
st.write(f"Predicted price for {ticker} in 30 days: ${predicted_price:.2f}")

# Investment advice based on user input
st.header("Investment Advice")
if investment_goal == "Growth":
    advice = "Consider investing in high-growth stocks or ETFs."
elif investment_goal == "Income":
    advice = "Consider investing in dividend-paying stocks or bonds."
else:
    advice = "Consider investing in stable assets like bonds or blue-chip stocks."

if risk_tolerance < 4:
    advice += " Given your low risk tolerance, a conservative approach is recommended."
elif risk_tolerance > 7:
    advice += " Given your high risk tolerance, you can consider more aggressive investments."

st.write(advice)


@st.cache_data(ttl=60*10)  # Cache data for 10 minutes
def get_real_time_data(ticker):
    return yf.Ticker(ticker).history(period="1d")
