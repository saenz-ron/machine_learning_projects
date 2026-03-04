# Author Ronnie Saenz

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Last four digits of USD ID - 6857
USD_digits = [6,8,5,7]

data_set = np.random.uniform(0,1, size=(100, 4))

# Finds the standard deviaton
std_dev = ((6+8+5+7)/4)/5

# Produces the guassian noise
col_5 = np.random.normal(0, std_dev, 100)

# col5 = (6 * col1) + (8 * col2) + (5 * col3) + (7 * col4)
for i in range(100):
    col_5[i] = (6 * data_set[i][0]) + (8 * data_set[i][1]) + (5 * data_set[i][2]) + (7 * data_set[i][3])


# start of manual linear regression
theta = np.linalg.inv(data_set.T @ data_set) @ data_set.T @ col_5



# start of linear regression with sklearn
y = 6 * data_set[:, 0] + 8 * data_set[:, 1] + 5 * data_set[:, 2] + 7 * data_set[:, 3]
# sklearn.linear_model.LinearRegression(*, fit_intercept = True, copy_X=True, n_jobs = None, positive = False)
reg = LinearRegression().fit(data_set,y)

# theta and reg need to be converted into arrays
theta_a = []
reg_a = []
for i in range(4):
    theta_a.append(theta[i])
    reg_a.append(reg.coef_[i])

# print the normal equation predicted coefficients
print("Predicted coefficient using the Normal Equation: " , theta_a)

# Print the regression coefficient
print("Predicted coefficient using LinearRegression(): " , reg_a)

print("Difference between Normal equation prediction and actual coefficients: " , np.subtract(USD_digits , theta_a))

print("Difference between LinearRegression() prediction and actual coefficients: " , np.subtract(USD_digits , reg_a))

print("Difference between Normal equation prediction and Normal equation prediction coefficients: " , np.subtract(reg_a , theta_a))

