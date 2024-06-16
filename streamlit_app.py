import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit as st

st.title('Bitcoin Price Prediction')

# Get user input for date range
start_date = st.date_input('Start Date', pd.to_datetime('2020-06-14'))
end_date = st.date_input('End Date', pd.to_datetime('2024-06-14'))

# Fetch data from Yahoo Finance
df = yf.download('BTC-USD', start=start_date, end=end_date)

st.subheader('Bitcoin Data')
st.write(df.head())

# Visualize the closing price history
st.subheader('Closing Price History')
fig, ax = plt.subplots(figsize=(16,8))
ax.plot(df['Close'])
ax.set_title('Close Price History')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price USD ($)')
st.pyplot(fig)

# Preprocess the data
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
st.subheader(f'Root Mean Squared Error: {rmse}')

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

st.subheader('Model Predictions')
fig2, ax2 = plt.subplots(figsize=(16,8))
ax2.plot(train['Close'])
ax2.plot(valid[['Close', 'Predictions']])
ax2.set_title('Model')
ax2.set_xlabel('Date')
ax2.set_ylabel('Close Price USD ($)')
ax2.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig2)

# Show the valid and predicted prices
st.subheader('Valid and Predicted Prices')
st.write(valid.tail())

# Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_test = []
# Append the past 60 days
X_test.append(last_60_days_scaled)
# Convert the X_test data set to a numpy array
X_test = np.array(X_test)
# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get the predicted scaled price
pred_price = model.predict(X_test)
# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
st.subheader('Predicted Price for Next Day')
st.write(pred_price)

try:
    btc_quote2 = yf.download('BTC-USD', start=end_date, end=end_date + pd.DateOffset(1))  # Adjust end date to ensure the data includes the entire day
    if btc_quote2.empty:
        raise ValueError("No data found for BTC-USD for the specified date.")
    st.subheader('Actual Price for Next Day')
    st.write(btc_quote2['Close'])
except Exception as e:
    st.error(f"Error downloading data: {e}")

