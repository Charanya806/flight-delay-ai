import pandas as pd

file_path = "data/raw/flights.csv"

print("Loading first 10 rows...")

df = pd.read_csv(file_path, nrows=10)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== SAMPLE SHAPE ==========")
print(df.shape)

print("\nReal dataset check completed successfully!")