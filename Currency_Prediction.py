import tensorflow as tf
import os
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt 
"""Ignore if there is any warnign """
warnings.filterwarnings('ignore')
"""Further steps of making app using streamlit"""
import streamlit as st

#Copy of the Data as a pandas DataFrame
# adding  path of  the dataset
filepath = r"C:/Users/baqer/OneDrive - OsloMet/Oslomet ACIT/Høst2023/ACIT4420 Problem Solvign with scripting/Final_project/02.Data/DataNorgesBank.xlsx"
# Loading the dataset as pandas DataFrame
raw_data = pd.read_excel(filepath)
df = raw_data.copy()

# Convertion of 'Time' column to datetime format 

df['Time'] = pd.to_datetime(df['Time'])

# Initiate the time_window_sizes list to store the time window size 
time_window_sizes = []
while True:
    try:
        value = int(input("Enter a time window size (choose from [3, 5, 10, 15,20,30, ...]): "))
        if value in [3, 5, 10, 15]:
            time_window_sizes.append(value)
        else:
            print("Invalid value. Please choose one of [3, 5, 10, 15,20,30,...,].")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

    response = input("Do you want to add another time window size? (yes/no): ").lower()
    if response != 'yes':
        break  # Exit the loop if the user does not want to add more time window sizes

# Now time_window_sizes contains the opted time window sizes
print("choosen time window sizes:", time_window_sizes)

# Initiate the cuurency to store the number of currency
currencies = []

# Ask the user for the number of currencies they want to predict
num_currencies = int(input("How many currencies do you want to forecast? Max:(12) Enter a number: "))

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
                print("Invalid currency. Please enter one of [HKD, EUR, CAD, USD, AUD, SGD, JPY, PKR, SEK, NZD, TRY, BRL].")
        except ValueError:
            print("Invalid value. Please try again.")

# Now 'currencies' list contains the selected currencies
print("Selected currencies:", currencies)

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['Currency', 'Time Window Size', 'Total Windows', 'Correct Predictions (Rule 1)',
                                  'Correct Predictions (Rule 2)', 'Correct Predictions (Rule 3)',
                                  'Correct Predictions (Rule 4)', 'Correct Predictions (Rule 5)',
                                  'Correct Predictions (Rule 6)', 'Percentage Correct (Rule 1)',
                                  'Percentage Correct (Rule 2)', 'Percentage Correct (Rule 3)',
                                  'Percentage Correct (Rule 4)', 'Percentage Correct (Rule 5)',
                                  'Percentage Correct (Rule 6)'])

# Iterate through the time window sizes and currencies
for time_window_size in time_window_sizes:
    for currency in currencies:
        # Create counters for correctly predicted segments for each rule
        correct_predictions_rule1 = 0
        correct_predictions_rule2 = 0
        correct_predictions_rule3 = 0
        correct_predictions_rule4 = 0
        correct_predictions_rule5 = 0
        correct_predictions_rule6 = 0

        total_windows = 0

        for i in range(0, len(df) - time_window_size + 1, time_window_size):
            window_data = df.iloc[i:i + time_window_size]
            segment_difference = window_data[currency].diff().iloc[-1]

            # Implement Rule 1 for prediction
            rule1_prediction = 'Positive' if segment_difference > 0 else 'Negative'
            if rule1_prediction == 'Positive' and segment_difference > 0:
                correct_predictions_rule1 += 1
            elif rule1_prediction == 'Negative' and segment_difference < 0:
                correct_predictions_rule1 += 1

            # Implement Rule 2 for prediction
            avg_currency = window_data[currency].mean()
            prev_avg_currency = df[currency].iloc[i - time_window_size:i].mean()
            rule2_prediction = 'Positive' if avg_currency > prev_avg_currency else 'Negative'
            if rule2_prediction == 'Positive' and segment_difference > 0:
                correct_predictions_rule2 += 1
            elif rule2_prediction == 'Negative' and segment_difference < 0:
                correct_predictions_rule2 += 1

            # Implement Rule 3 for prediction
            correlation = window_data[currency].corr(window_data['Time'])
            rule3_prediction = 'Positive' if correlation > 0 else 'Negative'
            if rule3_prediction == 'Positive' and segment_difference > 0:
                correct_predictions_rule3 += 1
            elif rule3_prediction == 'Negative' and segment_difference < 0:
                correct_predictions_rule3 += 1

            # Implement Rule 4 for prediction
            max_currency = window_data[currency].max()
            prev_max_currency = df[currency].iloc[i - time_window_size:i].max()
            rule4_prediction = 'Positive' if max_currency > prev_max_currency else 'Negative'
            if rule4_prediction == 'Positive' and segment_difference > 0:
                correct_predictions_rule4 += 1
            elif rule4_prediction == 'Negative' and segment_difference < 0:
                correct_predictions_rule4 += 1

            # Implement Rule 5 for prediction
            min_currency = window_data[currency].min()
            prev_min_currency = df[currency].iloc[i - time_window_size:i].min()
            rule5_prediction = 'Positive' if min_currency > prev_min_currency else 'Negative'
            if rule5_prediction == 'Positive' and segment_difference > 0:
                correct_predictions_rule5 += 1
            elif rule5_prediction == 'Negative' and segment_difference < 0:
                correct_predictions_rule5 += 1

            # Implement Rule 6 for prediction
            median_currency = window_data[currency].median()
            prev_median_currency = df[currency].iloc[i - time_window_size:i].median()
            rule6_prediction = 'Positive' if median_currency > prev_median_currency else 'Negative'
            if rule6_prediction == 'Positive' and segment_difference > 0:
                correct_predictions_rule6 += 1
            elif rule6_prediction == 'Negative' and segment_difference < 0:
                correct_predictions_rule6 += 1
            total_windows += 1

        # Calculate the percentage of correctly predicted segments for each rule
        percentage_correct_rule1 = (correct_predictions_rule1 / total_windows) * 100
        percentage_correct_rule2 = (correct_predictions_rule2 / total_windows) * 100
        percentage_correct_rule3 = (correct_predictions_rule3 / total_windows) * 100
        percentage_correct_rule4 = (correct_predictions_rule4 / total_windows) * 100
        percentage_correct_rule5 = (correct_predictions_rule5 / total_windows) * 100
        percentage_correct_rule6 = (correct_predictions_rule6 / total_windows) * 100

        # Create a temporary DataFrame for the current combination
        temp_df = pd.DataFrame({
            'Currency': [currency],
            'Time Window Size': [time_window_size],
            'Total Windows': [total_windows],
            'Correct Predictions (Rule 1)': [correct_predictions_rule1],
            'Correct Predictions (Rule 2)': [correct_predictions_rule2],
            'Correct Predictions (Rule 3)': [correct_predictions_rule3],
            'Correct Predictions (Rule 4)': [correct_predictions_rule4],
            'Correct Predictions (Rule 5)': [correct_predictions_rule5],
            'Correct Predictions (Rule 6)': [correct_predictions_rule6],
            'Percentage Correct (Rule 1)': [percentage_correct_rule1],
            'Percentage Correct (Rule 2)': [percentage_correct_rule2],
            'Percentage Correct (Rule 3)': [percentage_correct_rule3],
            'Percentage Correct (Rule 4)': [percentage_correct_rule4],
            'Percentage Correct (Rule 5)': [percentage_correct_rule5],
            'Percentage Correct (Rule 6)': [percentage_correct_rule6]
        })
      
        # Concatenate the temporary DataFrame with the result DataFrame
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

# Display the result DataFrame for all combinations of time window sizes and currencies

# Print the dataset
print(f"\nResult DataFrame:\n{result_df}")

"""Machine Learning Model (LSTM() """ 

    

# Ask the user whether to continue with machine learning
while True:
    try:
        continue_ml = input("Do you want to continue with machine learning (LSTM)? (yes/no): ").lower()

        if continue_ml == 'yes':
            # Refer to machine learning code 
            print("Continuing with machine learning by opening  (Machine_Learning_Currency_Prediction.py)")
            # ...
            break
        elif continue_ml == 'no':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")
    except ValueError:
        print("Invalid value. Please try again.")