import numpy as np

# Task: Perform broadcasting between different-shaped arrays
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

# Broadcasting addition
broadcast_sum = arr_2d + arr_1d
print("Broadcasting Addition:\n", broadcast_sum)

# Another broadcasting example
arr_col = np.array([[100], [200]])
broadcast_sum_col = arr_2d + arr_col
print("\nBroadcasting with a column array:\n", broadcast_sum_col)

# Task: Generate random arrays and apply statistical computations
# Generate a 3x4 random array
random_arr = np.random.rand(3, 4)
print("Random 3x4 Array:\n", random_arr)

# Mean of the entire array
print("\nMean of the array:", np.mean(random_arr))

# Standard deviation along columns
print("Standard deviation along columns:", np.std(random_arr, axis=0))

# Sum along rows
print("Sum along rows:", np.sum(random_arr, axis=1))

# Task: Implement linear algebra operations using numpy.linalg
# Create two matrices
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
matrix_product = np.dot(matrix_a, matrix_b)
print("Matrix Multiplication (using np.dot):\n", matrix_product)

# Using numpy.linalg.det() to find the determinant
determinant_a = np.linalg.det(matrix_a)
print("\nDeterminant of Matrix A:", determinant_a)

# Using numpy.linalg.inv() to find the inverse
inverse_a = np.linalg.inv(matrix_a)
print("\nInverse of Matrix A:\n", inverse_a)

# Solving a linear system Ax = b
# Let A be matrix_a and b = np.array([9, 10])
b = np.array([9, 10])
solution_x = np.linalg.solve(matrix_a, b)
print("\nSolution of Ax = b:", solution_x)

