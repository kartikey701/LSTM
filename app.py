import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

from datetime import date
import yfinance as yf

# Streamlit app title
st.title('Stock Trend Prediction')

# User input for the stock ticker symbol
user_input = st.text_input('Enter stock ticker', 'MSFT')

# Check if the user has entered a ticker symbol
if user_input:
    # Fetch historical stock data using yfinance
    df = yf.download(user_input, start='2012-01-01', end='2022-01-01')
    
    # Display the data description
    st.subheader('Data from 2012 - 2022')
    st.write(df.describe())

    # Visualization: Closing Price VS Time Chart
    st.subheader('Closing Price VS Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # Visualization: Closing Price VS Time Chart with 100 moving average(MA)
    st.subheader('Closing Price VS Time Chart with 100 moving average(MA)')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    # Visualization: Closing Price VS Time Chart with 100 and 200 moving average(MA)
    st.subheader('Closing Price VS Time Chart with 100 and 200 moving average(MA)')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)

    # Splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    # Load pre-trained model
    model = load_model('keras.h5')

    # Testing
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data = final_df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(input_data)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final graph: Prediction vs Original
    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original price')
    plt.plot(y_predicted, 'r', label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

