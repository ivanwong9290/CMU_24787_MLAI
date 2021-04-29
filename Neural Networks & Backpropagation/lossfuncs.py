#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:23:40 2021

@author: lbk
"""
import numpy as np # helps with the math
from sklearn.metrics import mean_squared_error

d = 2
n = 20

y_true = np.random.rand(2,20)
y_pred = np.random.rand(2,20)

mse = mean_squared_error(y_true, y_pred)

print(mse)

e = y_true - y_pred

L = np.matrix.trace(e @ e.T)

print(L/2/n)



