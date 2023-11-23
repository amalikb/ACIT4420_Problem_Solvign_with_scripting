# ACIT4420_Problem_Solving_with_Scripting

## Norwegian Currency Prediction Using Basic Rules and LSTM 

This repository contains scripts for predicting Norwegian currency using basic rules and LSTM (Long Short-Term Memory) models.

## Dataset Package

The project utilizes the dataset from Norges Bank.

The combined dataset is available in the "Dataset" folder.

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

• Download the exchange rates for some currencies, 10 years back.
• Divide the data in time windows and calculate the difference between the
start and end of the segment.
• Implement more than one rule to predict whether the difference between
the start and end in a given time window will be positive or negative, based
on the data only from previous time windows.
• Suggestion for a (simple) rule: if the correlation (between the time points
and the exchange rate) in the preceding window is positive, predict a positive
difference. If the correlation is negative, predict a negative difference.
• Output the percentage of correctly predicted time windows or segments.
• For a given currency, use the program to produce a table comparing the
prediction accuracy of all implemented prediction rules for 4 different time
windows. Make a table of comparison for at least two currencies, and
conclude regarding the prediction performance.
These are minimum features which must be implemented, but
