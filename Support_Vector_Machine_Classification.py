#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:13:13 2019

@author: gokhan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train, y_train)

print("Accuracy of SVM Algorithm: ", svm.score(x_test, y_test))










