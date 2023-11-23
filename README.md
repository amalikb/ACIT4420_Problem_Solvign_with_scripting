# ACIT4420_Problem_Solving_with_Scripting

## Norwegian Currency Prediction Using simple Rules and LSTM 

This repository contains scripts for predicting Norwegian currency using **basic rules** and **LSTM (Long Short-Term Memory)** models.

## Dataset Package

The project utilizes the dataset from Norges Bank.

The combined dataset is available in the **"Dataset"** folder.
Foreign currency list used in this project versus NOK is listed in table below 

| Numbers | Currency (13.10.2023) | Exchange Rate (NOK) |
|---------|------------------------|----------------------|
| 1       | 1 HKD                 | 1.40                 |
| 2       | 1 EUR                 | 11.52                |
| 3       | 1 CAD                 | 8.00                 |
| 4       | 1 USD                 | 10.94                |
| 5       | 1 AUD                 | 6.91                 |
| 6       | 1 SGD                 | 8.14                 |
| 7       | 1 JPY                 | 7.31                 |
| 8       | 100 PKR               | 3.94                 |
| 9       | 100 SEK               | 99.71                |
| 10      | 1 NZD                 | 6.46                 |
| 11      | 100 TRY               | 39.36                |
| 12      | 1 BRL                 | 2.16                 |


## Script Description



Please update the directory with the corresponding username in order to run the script successfully!

```python
# Python Package Installation
# Use the package manager pip to install essential packages if needed.
pip install numpy
pip install pandas
pip install tensorflow
pip install seaborn
pip install matplotlib
```
# Required libraries are imported into the Python script.
```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
```

# First phase is the the scripting based on simple rules 

- Download the exchange rates for some currencies, from 2013-1023

- The data divided in **time windows(3,5,10,15)** and difference between the start and end of the segment been calculated
- Six  Rules have been implemented using segment difference relationship with min,max,mean,median, correlation with time, and simple coparison of totale time window with segment difference
- Output the percentage of correctly predicted time windows is presented as pandas dataframe as well as barplot(availabe in **images** folder
- 

