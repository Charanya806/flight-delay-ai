import pandas as pd

# Load real flight dataset
df = pd.read_csv("data/raw/flights.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== BASIC STATISTICS ==========")
print(df.describe())

print("\n========== AIRLINES ==========")
print(df["AIRLINE"].value_counts())

print("\n========== CANCELLED FLIGHTS ==========")
print(df["CANCELLED"].value_counts())

print("\n========== DIVERTED FLIGHTS ==========")
print(df["DIVERTED"].value_counts())

print("\nData quality check completed successfully!")