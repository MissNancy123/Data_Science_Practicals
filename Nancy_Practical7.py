# 🧰 Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# -------------------------
# Step 1: Create a Sample Dataset
# -------------------------
data = {
    'Age': [25, 30, np.nan, 35, 40, np.nan],
    'Salary': [50000, 54000, 58000, np.nan, 62000, 65000],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', np.nan, 'Kolkata']
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# -------------------------
# Step 2: Handle Missing Data using SimpleImputer
# -------------------------
imputer_mean = SimpleImputer(strategy='mean')
df['Age'] = imputer_mean.fit_transform(df[['Age']])
df['Salary'] = imputer_mean.fit_transform(df[['Salary']])

imputer_most_frequent = SimpleImputer(strategy='most_frequent')
df['City'] = imputer_most_frequent.fit_transform(df[['City']]).ravel()  # ✅ fixed

print("\nAfter Handling Missing Values:\n", df)

# -------------------------
# Step 3: Encode Categorical Variables
# -------------------------
# Label Encoding
label_encoder = LabelEncoder()
df['City_Label'] = label_encoder.fit_transform(df['City'])

# One-Hot Encoding using pandas get_dummies
onehot = pd.get_dummies(df['City'], prefix='City')
df = pd.concat([df, onehot], axis=1)

print("\nAfter Encoding Categorical Variables:\n", df)

# -------------------------
# Step 4: Feature Scaling (Standardization and Normalization)
# -------------------------
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

df['Age_StandardScaled'] = scaler_standard.fit_transform(df[['Age']])
df['Salary_MinMaxScaled'] = scaler_minmax.fit_transform(df[['Salary']])

print("\nAfter Feature Scaling:\n", df)

# -------------------------
# Step 5: Split Dataset into Training and Testing Sets
# -------------------------
X = df[['Age', 'Salary', 'City_Label']]
y = df['Salary']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTraining Features:\n", X_train)
print("\nTesting Features:\n", X_test)

