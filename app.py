from tracemalloc import start
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import cufflinks as cf
from datetime import date, timedelta


st.title('Stock Trend Prediction')


st.sidebar.subheader('Time Input')
start_date = st.sidebar.date_input("Start date", date(2017, 1, 1))
end_date = st.sidebar.date_input("End date", (date.today()))


# Stock Input using inital date
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start_date, end_date)

tickerData = yf.Ticker(user_input)
tickerDF = tickerData.history(period='1d', start = start_date, end = end_date)

# Select ticker symbol
tickerData = yf.Ticker(user_input)

# TICKER INFORMATION

# Ticker Logo
st.header('\n')
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

# Ticker Regular Name
st.header('\n')
string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

# Ticker Summary
string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)



# GRAPHING

# Bollinger bands, upper and lower bounds of the stock prediction 
st.header('Bollinger Bands')
qf = cf.QuantFig(tickerDF, title='First Quant Figure', legend='top', name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

# Visualization of the graphs
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# How we scale the data
scaler = MinMaxScaler(feature_range=(0, 1))

# The array for data training
data_training_array = scaler.fit_transform(data_training)


# Loading my module
model = load_model('keras_model.h5')

# Testing my module
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

# How much we scale it as well
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Prediction vs Original model
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)