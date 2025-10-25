# ✅ Step 1: Import Required Library
import pandas as pd

# ✅ Step 2: Load the Dataset

df = pd.read_csv(r"C:\Users\HP\Downloads\Data.csv")

# Display first few rows
print("🔹 Original Dataset:")
print(df.head())

# ✅ Step 3: Check for Missing Values
print("\n🔹 Missing Values Before Cleaning:")
print(df.isnull().sum())

# ✅ Step 4: Handle Missing Values
# Fill missing numerical columns with mean
if 'Item_Weight' in df.columns:
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

# Fill missing categorical columns with mode
if 'Outlet_Size' in df.columns:
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

print("\n✅ Missing values handled successfully!")

# ✅ Step 5: Handle Duplicates
print("\n🔹 Duplicates Before:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

print("✅ Duplicates removed successfully!")
print("🔹 Duplicates After:", df.duplicated().sum())

# ✅ Step 6: Use groupby() for Aggregation
if 'Outlet_Type' in df.columns:
    print("\n🔹 Average Item_Weight by Outlet_Type:")
    print(df.groupby('Outlet_Type')['Item_Weight'].mean())

# ✅ Step 7: Use describe() for Statistical Summary
print("\n🔹 Dataset Summary:")
print(df.describe(include='all'))

# ✅ Step 8: Apply Filtering Techniques
if 'Item_MRP' in df.columns:
    print("\n🔹 Items with MRP greater than 200:")
    print(df[df['Item_MRP'] > 200])

if 'Outlet_Location_Type' in df.columns:
    print("\n🔹 Items sold in Tier 1 outlets:")
    print(df[df['Outlet_Location_Type'] == 'Tier 1'])

# ✅ Step 9: Save the Cleaned Data
df.to_csv(r"C:\Users\HP\Desktop\Data Science\Cleaned_Data.csv", index=False)
print("\n✅ Cleaned dataset saved as 'Cleaned_Data.csv'")

# ✅ Step 10: Display Final Data Preview
print("\n🔹 Final Cleaned Dataset Preview:")
print(df.head())
