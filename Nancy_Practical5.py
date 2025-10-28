import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.arange(1, 11)
y = x * 2
y2 = x ** 2

# ---- Line Plot ----
plt.figure(figsize=(6, 4))
plt.plot(x, y, color='blue', marker='o', label='y = 2x')
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()

# ---- Bar Plot ----
plt.figure(figsize=(6, 4))
plt.bar(x, y, color='green')
plt.title("Bar Plot")
plt.xlabel("X-axis")
plt.ylabel("Values")
plt.show()

# ---- Histogram ----
data = np.random.randn(1000)
plt.figure(figsize=(6, 4))
plt.hist(data, bins=20, color='purple', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# ---- Scatter Plot ----
plt.figure(figsize=(6, 4))
plt.scatter(x, y2, color='red', marker='*')
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y = X²")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(x, y, label='Linear (y=2x)', color='orange', linestyle='--', marker='d')
plt.plot(x, y2, label='Quadratic (y=x²)', color='blue', linestyle='-', marker='o')
plt.title("Customized Plot Example", fontsize=14, fontweight='bold')
plt.xlabel("X Values", fontsize=12)
plt.ylabel("Y Values", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Line Plot
axes[0, 0].plot(x, y, color='blue', marker='o')
axes[0, 0].set_title("Line Plot")

# Bar Plot
axes[0, 1].bar(x, y, color='green')
axes[0, 1].set_title("Bar Plot")

# Histogram
axes[1, 0].hist(data, bins=15, color='purple', edgecolor='black')
axes[1, 0].set_title("Histogram")

# Scatter Plot
axes[1, 1].scatter(x, y2, color='red', marker='*')
axes[1, 1].set_title("Scatter Plot")

# Adjust layout
plt.tight_layout()
plt.show()

