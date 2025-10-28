# 🧰 Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Step 1: Load Dataset
# -------------------------
# Example: Simple dataset of hours studied vs student scores
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Student_Score': [10, 20, 30, 40, 50, 60, 65, 78, 85, 95]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# -------------------------
# Step 2: Split Data
# -------------------------
X = df[['Hours_Studied']]
y = df['Student_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Step 3: Linear Regression Model
# -------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# -------------------------
# Step 4: Polynomial Regression Model
# -------------------------
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_poly_pred = poly_model.predict(X_poly)

# -------------------------
# Step 5: Evaluation Metrics
# -------------------------
mae = mean_absolute_error(y_test, y_pred_linear)
mse = mean_squared_error(y_test, y_pred_linear)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_linear)

print("\n📊 Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------
# Step 6: Visualization
# -------------------------
plt.figure(figsize=(10,5))

# Linear Regression Line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue')
plt.plot(X, linear_model.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Student Score')

# Polynomial Regression Curve
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue')
plt.plot(X, y_poly_pred, color='green')
plt.title('Polynomial Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Student Score')

plt.tight_layout()
plt.show()
