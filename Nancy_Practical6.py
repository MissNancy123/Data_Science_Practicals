import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = sns.load_dataset("iris")  # dataset for visualization

# 1. Heatmap (Correlation Heatmap)

plt.figure(figsize=(8, 6))
correlation = data.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Iris Dataset")
plt.show()

# -------------------------
# 2. Pairplot
# -------------------------
sns.pairplot(data, hue="species", palette="husl")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# -------------------------
# 3. Boxplot
# -------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(x="species", y="sepal_length", data=data, palette="Set2")
plt.title("Boxplot of Sepal Length by Species")
plt.show()

# -------------------------
# 4. Combine Multiple Seaborn Plots in One Figure (Subplots)
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatterplot
sns.scatterplot(ax=axes[0], data=data, x="sepal_length", y="sepal_width", hue="species", palette="cool")
axes[0].set_title("Scatterplot")

# Boxplot
sns.boxplot(ax=axes[1], data=data, x="species", y="petal_length", palette="pastel")
axes[1].set_title("Boxplot")

# Violinplot
sns.violinplot(ax=axes[2], data=data, x="species", y="petal_width", palette="muted")
axes[2].set_title("Violinplot")

plt.suptitle("Multiple Seaborn Plots Combined", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


