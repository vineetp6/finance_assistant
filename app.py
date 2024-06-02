import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from scipy.optimize import minimize

# Function to fetch stock data
@st.cache_data(ttl=60*10)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    return hist

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# ARIMA model
def arima_model(data):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def predict_arima(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

# LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, x_train, y_train, epochs=20, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def create_dataset(data, time_step=60):
    x, y = [], []
    for i in range(len(data)-time_step-1):
        x.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

def predict_lstm(model, data, scaler, time_step=60):
    x_input = data[-time_step:].reshape(1, -1)
    temp_input = list(x_input[0])
    lst_output = []
    n_steps = time_step
    i = 0
    while(i < 30):
        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            lst_output.append(yhat[0][0])
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i += 1
    return scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# Modern Portfolio Theory (MPT)
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_var = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_ret - risk_free_rate) / p_var

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

# Streamlit App
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

# Preprocess data for models
scaled_data, scaler = preprocess_data(stock_data['Close'].values.reshape(-1, 1))

# ARIMA prediction
st.header("ARIMA Model Prediction")
arima_model_fit = arima_model(stock_data['Close'])
arima_forecast = predict_arima(arima_model_fit, steps=30)
st.write(f"ARIMA model predicts next 30 days: {arima_forecast}")

# LSTM prediction
st.header("LSTM Model Prediction")
x_train, y_train = create_dataset(scaled_data)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
lstm_model = create_lstm_model((x_train.shape[1], 1))
trained_lstm_model = train_lstm_model(lstm_model, x_train, y_train)
lstm_forecast = predict_lstm(trained_lstm_model, scaled_data, scaler)
st.write(f"LSTM model predicts next 30 days: {lstm_forecast}")

# MPT portfolio optimization
st.header("Portfolio Optimization using MPT")
mean_returns = stock_data['Close'].pct_change().mean()
cov_matrix = stock_data['Close'].pct_change().cov()
optimal_portfolio = optimize_portfolio(mean_returns, cov_matrix)
st.write(f"Optimal portfolio: {optimal_portfolio}")

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
