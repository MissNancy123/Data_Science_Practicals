import numpy as np

# Task: Create 1D, 2D arrays
# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr_1d)

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:\n", arr_2d)

# Task: Perform indexing and slicing
# Indexing 1D array
print("Element at index 2 (1D):", arr_1d[2])

# Slicing 1D array
print("Elements from index 1 to 3 (1D):", arr_1d[1:4])

# Indexing 2D array
print("\nElement at row 1, column 2 (2D):", arr_2d[1, 2])

# Slicing 2D array (first two rows, all columns)
print("First two rows (2D):\n", arr_2d[:2, :])

# Slicing 2D array (all rows, first two columns)
print("First two columns (2D):\n", arr_2d[:, :2])

# Task: Perform element-wise and matrix operations
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Element-wise addition
print("Element-wise addition:\n", arr1 + arr2)

# Element-wise multiplication
print("\nElement-wise multiplication:\n", arr1 * arr2)

# Matrix multiplication (dot product)
print("\nMatrix multiplication:\n", np.dot(arr1, arr2))

# Task: Use functions like np.mean(), np.std(), np.dot()
data = np.array([10, 20, 30, 40, 50])

# Mean
print("Mean of data:", np.mean(data))

# Standard deviation
print("Standard deviation of data:", np.std(data))

# Dot product was already demonstrated in the previous cell, but here's another example
vec1 = np.array([1, 2])
vec2 = np.array([3, 4])
print("\nDot product of vec1 and vec2:", np.dot(vec1, vec2))