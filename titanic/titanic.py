"""
titanic.py - This file implements a model to use the Titanic passenger data 
(name, age, price of ticket, etc) to try to predict who will survive and who will die.

Author: Gautam Muralidharan
Created on: 2024-11-05
Last modified: 2024-11-05
License: <TBD>

Usage: python titanic.py [options]
Dependencies: NumPy, Pandas

This file contains:

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os

# Specify the path to the input folder
input_folder = os.path.join(os.path.dirname(__file__), 'input')

for dirname, _, filenames in os.walk(input_folder):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# load training data
train_data = pd.read_csv(input_folder+"/train.csv")
print(train_data.head())

# sns.pairplot(train_data)
# plt.show()

plt.figure(figsize=(12, 5))

# Pclass vs Survived
plt.subplot(1, 2, 1)
sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.title('Survival Count by Pclass')

# Sex vs Survived
plt.subplot(1, 2, 2)
sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.title('Survival Count by Sex')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.countplot(x='Embarked', hue='Survived', data=train_data)
plt.title('Survival Count by Embarked Port')
plt.show()
