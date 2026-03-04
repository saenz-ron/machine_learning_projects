# Author Ronnie Saenz
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, TruncatedSVD

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


''' Center the dataset '''

mean_values = np.mean(data_set, axis=0)
std_dev = np.std(data_set, axis=0)

# Subtract the mean from each feature
centered_data = (data_set - mean_values)


''' Perform PCA (by hand) '''

# Compute the covariance matrix
cov_matrix = np.dot(centered_data.T, centered_data) / (19)
    
# Obtains the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

''' Determine and display the variance '''

print("Variance calculated by hand: \n", eigenvalues / np.sum(eigenvalues) )


''' Perform sklearn PCA '''

# Instantiate PCA
pca = PCA()

# Fit PCA to data
pca.fit(centered_data)

# Transform data
transformed_data = pca.transform(centered_data)

# Print explained variance ratio
print("Variance calculated by PCA: \n", pca.explained_variance_ratio_)


'''Perform sklearn SVD '''

# Initiate SVD
U, S, VT = np.linalg.svd(centered_data, full_matrices=False)
explained_variance = S**2 / (centered_data.shape[0] - 1)
variance_proportional_svd = explained_variance / np.sum(explained_variance)


# Print explained variance ratio
print("Variance calculated by SVD: \n", variance_proportional_svd)

print("\n")

# Show that each of these methods produce virtually identical singular values

# Print the singular values which is the square root of the eigenvalues

singular_values_hand = np.sqrt(eigenvalues) * np.sqrt(19)

print("The singular values from PCA done by hand: \n", singular_values_hand)
print("The singular values from PCA: \n", pca.singular_values_)
print("The singular values from SVD: \n", S)