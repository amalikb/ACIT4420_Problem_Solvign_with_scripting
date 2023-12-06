"""Loading the libraries"""
import tensorflow as tf
import os
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt 
"""Ignore unimportant warnigns """
warnings.filterwarnings('ignore')
""" loading streamlit for furhter development of the app"""
import streamlit as st
# Adding path to the data 
filepath = r"C:/Users/baqer/OneDrive - OsloMet/Oslomet ACIT/Høst2023/ACIT4420 Problem Solvign with scripting/Final_project/02.Data/DataNorgesBank.xlsx"
# Loading the dataset as pandas DataFrame
raw_data = pd.read_excel(filepath)
# Compying the dataset for data preprocessing 
df = raw_data.copy()
# Setting the Time as datetime format
df['Time'] = pd.to_datetime(df['Time'], dayfirst=False)
# Set the 'Time' column as the index
df.set_index('Time', inplace=True)
# Displaying few samples of the dataset
df.head(2) # Two samples from first two rows
df.tail(2) # Two samples from last two rows
# Concat both as one single dataframe 
Sample = pd.concat([df.head(2), df.tail(2)])
# Displaying Sample as pandas DataFrame 
df.head(10)

# Pricincple of LSTM work
# [[[1], [2], [3], [4], [5]]] [6]
# [[[2], [3], [4], [5], [6]]] [7]
# [[[3], [4], [5], [6], [7]]] [8]
# taking a copy of the original dataset 
data = raw_data.iloc[:,1:13]
from sklearn.preprocessing import MinMaxScaler
# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the data and transform the data
scaled_data = scaler.fit_transform(data)

# Convert the scaled data back to a DataFrame
data1 = pd.DataFrame(scaled_data, columns=df.columns)
# Converting the pandas dataframe to numpy with a window size
def df_to_X_y(data1, window_size):
  df_as_np = data1.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

# Assing values to the variables and applying the function
# Enter the window size 
WINDOW_SIZE = int(input('Enter the window size:'))
"""Enter number of epochs"""
EPOCHS = int(input("Please enter number of epochs: "))
  #Initialize a list to store the selected currencies
currencies = []

# Ask the user for the number of currencies they want to predict
num_currencies = int(input("How many currencies do you want to forecast? Max:(1) Enter a number: "))

# Allow the user to input currencies
for i in range(num_currencies):
    while True:
        try:
            currency_input = input(
                f"Enter currency {i + 1} vs NOK ('HKD', 'EUR', 'CAD', 'USD', 'AUD', 'SGD', 'JPY', 'PKR', 'SEK','NZD', 'TRY', 'BRL'): "
            )
            if currency_input.upper() in ['HKD', 'EUR', 'CAD', 'USD', 'AUD', 'SGD', 'JPY', 'PKR', 'SEK','NZD', 'TRY', 'BRL']:
                currencies.append(currency_input.upper())
                break
            else:
                print("Invalid c1urrency. Please enter one of [HKD, EUR, CAD, USD, AUD, SGD, JPY, PKR, SEK, NZD, TRY, BRL].")
        except ValueError:
            print("Invalid value. Please try again.")

# Now 'currencies' list contains the selected currencies
print("Selected currencies:", currencies)

# Select the columns corresponding to the selected currencies
selected_columns = data[currencies]
selected_columns

X, y = df_to_X_y(selected_columns, WINDOW_SIZE)
X.shape, y.shape

# %%
X_train, y_train = X[:1500], y[:1500]
X_test, y_test = X[1500:], y[1500:]
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
# Loading the required libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, SGD

modellstm = Sequential()
modellstm.add(InputLayer((WINDOW_SIZE, 1)))
modellstm.add(LSTM(64))
modellstm.add(Dense(8, 'relu'))
modellstm.add(Dense(1, 'linear'))

# %%
#Compiling the model with indices loss, optimizer and learning reate 
modellstm.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[MeanAbsoluteError()])


history = modellstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS)
# Extract relevant metrics from the history object
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

train_loss = history.history['loss']
test_loss = history.history['val_loss']
mse = [mean_squared_error(y_test, modellstm.predict(X_test))]
mae = [mean_absolute_error(y_test, modellstm.predict(X_test))]
r_squared = r2_score(y_test, modellstm.predict(X_test))


# Plot the training and test loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')
plt.show()

# Print the MSE and MAE
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f"R-squared: {r_squared}")

# %%
train_predictions = modellstm.predict(X_train).flatten(); y_train_flatten = y_train.flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train_flatten})
train_results

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
# Plot the train predictions and actual values
plt.plot(train_results['Train Predictions'][:1000], label='Train Predictions',color='blue')
plt.plot(train_results['Actuals'][:1000], label='Actuals',color='black')
# Add a legend
plt.legend()
# Display the plot
plt.show()


# %%
test_predictions = modellstm.predict(X_test).flatten()
y_test_flatten = y_test.flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test_flatten})
test_results

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
# Assuming you have test_results DataFrame with columns 'Test Predictions' and 'Actuals'

plt.plot(test_results['Test Predictions'][:2000], label='Test Predictions',color='blue')
plt.plot(test_results['Actuals'][:2000], label='Actuals',color='black')
plt.legend()
plt.show()