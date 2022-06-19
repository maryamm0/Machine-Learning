#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:02:37 2022

@author: maryammanjoo
"""

## LINEAR REGRESSION
#Training Example feature value -> x_train
#Training Example targets -> y_train
#Training Example -> x_i, y_i
# m: Number of training examples	m
#parameter: weight,	w
#parameter: bias,b
#The result of the model evaluation at  𝑥(𝑖) 𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏 -> f_wb

import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0] #returns tuple of size of x_train
print(f"Number of training examples is: {m}")


#Method 2 to find training examples
m = len(x_train)
print(f"Number of training examples is: {m}")


i=0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
#for i in range(0,2):
 #   x_i = x_train[i]
  #  y_i = y_train[i]
   # print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

#Univariate linear regression function is:f=mx+b
#set w and b 
w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

#Creating univariate linear regression function
def lin_reg_model(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)  #returns an array of zeroes of size m
    for i in range(m):
        f_wb[i] = w * x[i] + b #linear regression formula for each training example
        
    return f_wb

first_f_wb = lin_reg_model(x_train, w, b)
print(first_f_wb)

# Plot model 
plt.plot(x_train, first_f_wb, c='b',label='Prediction regression')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Prices")
# Set the y-axis label
plt.ylabel('Price')
# Set the x-axis label
plt.xlabel('Size')
plt.legend()
plt.show()
