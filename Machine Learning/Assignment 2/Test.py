import numpy as np

# 2x3 matrix A
A = np.array([[1, 2, 3],
              [3, 2, 1]])

print("Start with matrix A: \n", A, '\n')

# Step 1: Compute the transpose of A
A_transpose = A.T
print("Step 1: Compute the transpose of A \n", A_transpose, '\n')

# Step 2: Multiply A by its transpose
A_times_AT = np.dot(A, A_transpose)
print("Step 2: Multiply A by its transpose \n", A_times_AT, '\n')

# Step 3: Multiply the transpose of A by A
AT_times_A = np.dot(A_transpose, A)
print("Step 3: Multiply the transpose of A by A \n", AT_times_A, '\n')

# Step 4: Take the square root of the results of step 2 and step 3
A_square = np.sqrt(A_times_AT)
B_square = np.sqrt(AT_times_A)
print("Step 4: Take the square root of the results of step 2 and step 3 \nA: ", A_square, "\nB: ", B_square,'\n')

# Display the square matrices
print("Square Matrix A_square:")
print(A_square)
print("\nSquare Matrix B_square:")
print(B_square)

