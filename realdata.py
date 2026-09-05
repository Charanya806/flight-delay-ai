import pandas as pd

# Load full real dataset
df = pd.read_csv("data/raw/flights.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== AIRLINE COUNT ==========")
print(df["AIRLINE"].value_counts())

print("\n========== CANCELLED FLIGHTS ==========")
print(df["CANCELLED"].value_counts())

print("\n========== DIVERTED FLIGHTS ==========")
print(df["DIVERTED"].value_counts())

print("\n========== DELAY STATISTICS ==========")
print(df[["DEPARTURE_DELAY", "ARRIVAL_DELAY"]].describe())

print("\n========== DATA QUALITY CHECK ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nData quality check completed successfully!")