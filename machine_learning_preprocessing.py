# ------------------------------------
# DATA PREPROCESSING
# Dataset : Bank Marketing
"""
This script performs data reading, cleaning, preprocessing,
encoding, and scaling on the Bank Marketing dataset.
"""
# ------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------
# Display settings
# -------------------------
pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)

# -------------------------
# 1. READ DATASET
# -------------------------
df = pd.read_csv("bank_marketing.csv")

# Strip column names
df.columns = df.columns.str.strip()

print("Original DataFrame Head:")
print(df.head())

# -------------------------
# 2. BASIC DATAFRAME DETAILS
# -------------------------
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("Column names:", df.columns.tolist())

print("Data types:")
print(df.dtypes)

print("DataFrame Info:")
df.info()

print("Statistical Description:")
print(df.describe())

# -------------------------
# 3. DATA READING TECHNIQUES
# -------------------------
print("Single column 'age':")
print(df['age'].head())

print("Multiple columns ['age','duration']:")
print(df[['age', 'duration']].head())

print("First 5 rows:")
print(df.head())

print("Last 5 rows:")
print(df.tail())

print("Row slicing 10-15:")
print(df[10:16])

print("Using loc:")
print(df.loc[10:15, ['age', 'job']])

print("Using iloc:")
print(df.iloc[10:15, 1:4])

# -------------------------
# 4. NULL VALUE CHECK
# -------------------------
print("Null values in each column:")
print(df.isna().sum())

# -------------------------
# 5. CREATE COPY FOR CLEANING
# -------------------------
new_df = df.copy()

# -------------------------
# 6. CLEAN CATEGORICAL COLUMNS
# -------------------------
categorical_columns = new_df.select_dtypes(include=['object']).columns.tolist()
numeric_columns = new_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Strip spaces and lowercase categorical values
for col in categorical_columns:
    new_df[col] = new_df[col].str.strip().str.lower()

# Print unique values for each categorical column
for col in categorical_columns:
    print(f"Unique values in '{col}':")
    print(new_df[col].unique())
    print("-" * 40)

# -------------------------
# 7. HANDLE MISSING VALUES (DEFENSIVE)
# -------------------------
for col in categorical_columns:
    new_df[col] = new_df[col].fillna('unknown')

for col in numeric_columns:
    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    new_df[col] = new_df[col].fillna(new_df[col].median())

# -------------------------
# 8. DUPLICATE REMOVAL
# -------------------------
print("Duplicate rows count:", new_df.duplicated().sum())
new_df = new_df.drop_duplicates()

# -------------------------
# 9. FEATURE ENGINEERING
# -------------------------
if 'age' in new_df.columns:
    new_df['age_group'] = pd.cut(
        new_df['age'],
        bins=[18, 30, 45, 60, 100],
        labels=['young', 'adult', 'senior', 'old']
    )

# -------------------------
# 10. OUTLIER HANDLING USING IQR (NUMERIC ONLY)
# -------------------------
print("Before Outlier Treatment (min & max):")
for col in numeric_columns:
    print(f"{col} -> min: {new_df[col].min()}, max: {new_df[col].max()}")

for col in numeric_columns:
    Q1 = new_df[col].quantile(0.25)
    Q3 = new_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    new_df[col] = np.where(
        new_df[col] < lower_bound, lower_bound,
        np.where(new_df[col] > upper_bound, upper_bound, new_df[col])
    )

print("Outliers treated using IQR method")

print("\nAfter Outlier Treatment (min & max):")
for col in numeric_columns:
    print(f"{col} -> min: {new_df[col].min()}, max: {new_df[col].max()}")

# -------------------------
# 11. ENCODING CATEGORICAL COLUMNS
# -------------------------
label_encoder = LabelEncoder()

# Binary categorical columns
binary_columns = [col for col in categorical_columns
                  if new_df[col].nunique() == 2]
for col in binary_columns:
    new_df[col] = label_encoder.fit_transform(new_df[col])

# Multi-class categorical columns
multi_class_columns = [col for col in categorical_columns
                       if new_df[col].nunique() > 2]
new_df = pd.get_dummies(new_df, columns=multi_class_columns, drop_first=True)

# -------------------------
# 12. FEATURE SCALING (Exclude target if present)
# -------------------------
numeric_columns_scaled = [col for col in numeric_columns
                          if col in new_df.columns]

scaler = StandardScaler()
new_df[numeric_columns_scaled] = scaler.fit_transform(new_df[numeric_columns_scaled])

# -------------------------
# 13. FINAL PREPROCESSED DATA SHAPE
# -------------------------
print("Number of rows:", new_df.shape[0])
print("Number of columns:", new_df.shape[1])

# -------------------------
# 14. SAVE PREPROCESSED DATA
# -------------------------
new_df.to_csv("bank_marketing_preprocessed.csv", index=False)
print("Preprocessed dataset saved successfully")
