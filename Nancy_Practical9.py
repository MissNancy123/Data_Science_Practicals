import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# -------------------------
# Step 1: Load Dataset
# -------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("🔹 First 5 Rows of Dataset:")
print(df.head())

# -------------------------
# Step 2: Split Data
# -------------------------
X = df.iloc[:, :-1]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Step 3: Train Models
# -------------------------
# Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# -------------------------
# Step 4: Evaluation Function
# -------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 {name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1-Score:", f1_score(y_true, y_pred, average='macro'))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Heatmap for confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# -------------------------
# Step 5: Evaluate All Models
# -------------------------
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("K-Nearest Neighbors (KNN)", y_test, y_pred_knn)
evaluate_model("Decision Tree", y_test, y_pred_tree)
