# Author Ronnie Saenz
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# data_set is a matrix that is M x 5 (M = 20)
# The first two columns of data_set are uniformally distributed between 0 and 10
# Columns 3-5 are determined as follows (A = 7, B = 5):
# 3rd column = Col1 + Col2
# 4th column = A Col1 + B Col2
# 5th column = B Col1 + A Col2

# Some variables
M = 10
A = 7
B = 5

# A seed to help with testing
np.random.seed(123)

col_1 = np.random.uniform(0,10, size=(M, 1))
col_2 = np.random.uniform(0,10, size=(M, 1))

noise = np.random.normal(0, 1.1, size=(M,1))

# columns 3-5 are set up as a 1D array 
col_3 = col_1 + col_2 + noise
col_4 = A * col_1 + B * col_2 + noise
col_5 = B * col_1 + A * col_2 + noise

# Concatenate the columns horizontally
data_set = np.hstack((col_1, col_2, col_3, col_4, col_5))

print("Combined Matrix (Horizontally):")
print(data_set)

# SVD Time
# Find AT * A
ATA = np.matmul(data_set.T, data_set)

# Gets the eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(ATA)

# Sorts the eigenvalues and eigenvectors
sorted_indices = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_indices]


# Compute singular values
singular_values = np.sqrt(abs(eigen_values))
sigma = np.zeros((20,5))
for i in range(len(singular_values)):
    sigma[i,i] = singular_values[i]

# Find V and VT
V = eigen_vectors
VT = eigen_vectors.T

# Inverting sigma
sigma_inv = np.zeros((5,20))
for i in range(5):
    sigma_inv[i,i] = 1/sigma[i,i]

# Find U
AV = np.matmul(data_set, V)
U = np.matmul(AV, sigma_inv)

# Asks the user fot the value of k
k = int(input("input your k value (0,1,2,3): "))

sigma_reduced = sigma[:5-k, :5-k]

# computing the reduced version
data_set_reduced = np.dot(U[:, :5-k], np.dot(sigma_reduced, VT[:5-k, :]))

relative_erorrs = np.abs(data_set - data_set_reduced) / np.abs(data_set)
average_relative_error = np.mean(relative_erorrs)

print("\nReduced Sigma Matrix:")
print(sigma_reduced)

print("\nAverage Relative Error:", average_relative_error)