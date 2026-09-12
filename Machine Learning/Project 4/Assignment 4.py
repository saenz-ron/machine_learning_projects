'''
Name: Ronnie Saenz
Assignment 4

Notes:
Let A be an integer whose value is the last digit of 
your USD ID#. Let B be the next-to-last digit. 
[A=7, B=5]


'''

import numpy as np
import matplotlib.pyplot as plt

# Number of points in the cloud
n = 100
A = 7
B = 5

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

col_1 = np.random.uniform(A, 2*A, n)
col_2 = np.random.uniform(B, 2*B, n)
col_3 = col_1 + col_2
col_4 = col_1 + (2*col_2)
col_5 = (2*col_1) + col_2

# Generate Gaussian noise
# gaussian_noise = np.random.normal(mean, std_dev, size)

gauss_col_3 = np.random.normal(0, .1 * np.mean(col_3), n)
gauss_col_4 = np.random.normal(0, .5 * np.mean(col_4), n)
gauss_col_5 = np.random.normal(0, np.mean(col_5), n)

col_3 += gauss_col_3
col_4 += gauss_col_4
col_5 += gauss_col_5

# Generate and display 3D cloud of random points in U(0, 1) cube
ax.scatter(col_3, col_4, col_5, marker='o', color='red')

ax.set_xlabel('col_3 Label')
ax.set_ylabel('col_4 Label')
ax.set_zlabel('col_5 Label')
ax.set_title('Assignment 4 Synthetic Dataset')
plt.show()

# Compute and display the mean and standard deviation of each column (dimension)
print("Compute and display the mean and standard deviation of each column (dimension)")
print("col_3 (mean, std deviation): " , np.mean(col_3) , "," , np.std(col_3))
print("col_4 (mean, std deviation): " , np.mean(col_4) , "," , np.std(col_4))
print("col_5 (mean, std deviation): " , np.mean(col_5) , "," , np.std(col_5))

# Center and standardize all data (all columns).
centered_standardized_col_1 = (col_1 - np.mean(col_1)) / np.std(col_1)
centered_standardized_col_2 = (col_2 - np.mean(col_2)) / np.std(col_2)
centered_standardized_col_3 = (col_3 - np.mean(col_3)) / np.std(col_3)
centered_standardized_col_4 = (col_4 - np.mean(col_4)) / np.std(col_4)
centered_standardized_col_5 = (col_5 - np.mean(col_5)) / np.std(col_5)

# Compute and display the mean and standard deviation of each column after centering and standardization.
print("\nCompute and display the mean and standard deviation of each column after centering and standardization")
print("col_1 (mean, std deviation): " , np.mean(centered_standardized_col_1) , "," , np.std(centered_standardized_col_1))
print("col_2 (mean, std deviation): " , np.mean(centered_standardized_col_2) , "," , np.std(centered_standardized_col_2))
print("col_3 (mean, std deviation): " , np.mean(centered_standardized_col_3) , "," , np.std(centered_standardized_col_3))
print("col_4 (mean, std deviation): " , np.mean(centered_standardized_col_4) , "," , np.std(centered_standardized_col_4))
print("col_5 (mean, std deviation): " , np.mean(centered_standardized_col_5) , "," , np.std(centered_standardized_col_5))
