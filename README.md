# ACIT4420_Problem_Solving_with_Scripting

## Norwegian Currency Prediction Using simple Rules and LSTM 

This repository contains scripts for predicting Norwegian currency using **basic rules** and **LSTM (Long Short-Term Memory)** models.

## Dataset Package

The project utilizes the dataset from Norges Bank.

The combined dataset is available in the **"Dataset"** folder.

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

