import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")

# Step 2: Basic overview
print("----- First 5 Rows -----")
print(df.head())

print("\n----- Dataset Info -----")
print(df.info())

print("\n----- Summary Statistics -----")
print(df.describe())

# Step 3: Check for missing values
print("\n----- Missing Values -----")
print(df.isnull().sum())

# Step 4: Categorical value counts
print("\n----- Value Counts for Categorical Columns -----")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\n{col}:\n{df[col].value_counts()}")

# Step 5: Correlation matrix
print("\n----- Correlation Matrix -----")
corr = df.corr(numeric_only=True)
print(corr)

# Step 6: Clean Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f"
)
plt.title("Correlation Heatmap (Numeric Columns Only)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 7: Value count visualization
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df, palette='viridis')
plt.title("Distribution of Students by Gender", fontsize=12)
plt.show()

# Step 8: Pivot table
print("\n----- Pivot Table: Average Marks by Gender and Stream -----")
pivot = df.pivot_table(values='Marks', index='Stream', columns='Gender', aggfunc='mean')
print(pivot)

# Step 9: Optional scatter plot for trend
plt.figure(figsize=(6,4))
sns.scatterplot(x='Hours_Studied', y='Marks', hue='Gender', data=df)
plt.title("Marks vs Hours Studied", fontsize=12)
plt.show()
