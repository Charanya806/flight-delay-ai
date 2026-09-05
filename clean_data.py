import pandas as pd
import os

# Load raw dataset
df = pd.read_csv("data/raw/flights.csv")

print("Original shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Convert delay columns to numeric
delay_columns = [
    "DEPARTURE_DELAY",
    "ARRIVAL_DELAY",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

for col in delay_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill delay-reason missing values with 0
reason_columns = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

df[reason_columns] = df[reason_columns].fillna(0)

# Create processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
output_path = "data/processed/flights_cleaned.csv"
df.to_csv(output_path, index=False)

print("Cleaned shape:", df.shape)
print("Saved to:", output_path)
print("\nData cleaning completed successfully!")