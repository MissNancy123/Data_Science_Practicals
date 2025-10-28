# 🔹 Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Step 1: Load Dataset
# -------------------------
iris = load_iris()
X = iris.data
y = iris.target

print("🔹 Dataset Shape:", X.shape)
print("🔹 Target Classes:", iris.target_names)

# -------------------------
# Step 2: Split Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Step 3: Define Models
# -------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# -------------------------
# Step 4: K-Fold Cross Validation
# -------------------------
print("\n📊 Cross-Validation Results:")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

# -------------------------
# Step 5: Hyperparameter Tuning with GridSearchCV
# -------------------------

# KNN Hyperparameter Tuning
knn_params = {'n_neighbors': [3, 5, 7, 9]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_train, y_train)

# Decision Tree Hyperparameter Tuning
tree_params = {'max_depth': [2, 3, 4, 5, None], 'criterion': ['gini', 'entropy']}
tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), tree_params, cv=5)
tree_grid.fit(X_train, y_train)

# -------------------------
# Step 6: Evaluate Best Models
# -------------------------
best_knn = knn_grid.best_estimator_
best_tree = tree_grid.best_estimator_

print("\n🏆 Best Parameters Found:")
print("KNN:", knn_grid.best_params_)
print("Decision Tree:", tree_grid.best_params_)

# Predict using the best models
y_pred_knn = best_knn.predict(X_test)
y_pred_tree = best_tree.predict(X_test)

print("\n📈 Model Performance Comparison:")
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

# Detailed Classification Reports
print("\n🔹 KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("\n🔹 Decision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
