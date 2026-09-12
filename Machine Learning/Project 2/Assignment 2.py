# Eigen decomposition

import numpy as np

# Define the 2x3 matrix A
A = np.array([[1, 2, 3],
              [3, 2, 1]])

# Step 1: Compute A^T * A
ATA = np.dot(A.T, A)
print("A^T * A:")
print(ATA)

# Step 2: Compute the eigenvalues (λ) and eigenvectors (V) of A^T * A
eigenvalues, eigenvectors = np.linalg.eig(ATA)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Step 3: Construct the diagonal matrix D from the eigenvalues
D = np.diag(eigenvalues)
print("\nDiagonal matrix D:")
print(D)

# Step 4: Compute the matrix P from the eigenvectors
P = eigenvectors
print("\nMatrix P:")
print(P)

# Step 5: Compute the inverse of P
P_inv = np.linalg.inv(P)
print("\nInverse of P:")
print(P_inv)

# Verify that the product of P, D, and P_inv is equal to the original matrix A
reconstructed_A = np.dot(np.dot(P, D), P_inv)
print("\nReconstructed Matrix A:")
print(reconstructed_A)

# If you want to obtain a second square matrix, let's call it B
# You can perform SVD on A to get U, Sigma, Vt
U, Sigma, Vt = np.linalg.svd(A)
B = np.dot(U, np.dot(np.diag(Sigma), Vt))
print("\nMatrix B (Reconstructed using SVD):")
print(B)
