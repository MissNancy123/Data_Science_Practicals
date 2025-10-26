import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Load Dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")
print("✅ Dataset Loaded Successfully!\n")
print("🔹 First 5 Rows:")
print(df.head())

# 2️⃣ Basic Info and Missing Values
print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Missing Values:")
print(df.isnull().sum())

# 3️⃣ Summary Statistics
print("\n🔹 Summary Statistics:")
print(df.describe(include='all'))

# 4️⃣ Categorical Variable Analysis
print("\n🔹 Value Counts for Categorical Columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}:\n", df[col].value_counts())

# 5️⃣ Correlation Analysis (numeric columns)
corr = df.corr(numeric_only=True)
print("\n🔹 Correlation Matrix:\n", corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Columns Only)")
plt.show()

# 6️⃣ Pivot Table Example
pivot = df.pivot_table(values='Fare', index='Pclass', columns='Sex', aggfunc='mean')
print("\n🔹 Pivot Table (Average Fare by Class and Gender):\n", pivot)

# 7️⃣ Filtering Examples
print("\n🔹 Passengers older than 50:\n", df[df['Age'] > 50].head())
print("\n🔹 Female Passengers in First Class:\n", df[(df['Sex'] == 'female') & (df['Pclass'] == 1)].head())

# 8️⃣ Visualization Examples
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

sns.histplot(df['Age'].dropna(), bins=20, kde=True)
plt.title("Age Distribution")
plt.show()


