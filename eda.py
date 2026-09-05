import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
df = pd.read_csv("data/processed/flights_cleaned.csv", low_memory=False)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Flights by Airline
airline_counts = df["AIRLINE"].value_counts()

plt.figure(figsize=(10, 6))
airline_counts.plot(kind="bar")
plt.title("Number of Flights by Airline")
plt.xlabel("Airline")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig("outputs/flights_by_airline.png")
plt.show()

# 2. Average Arrival Delay by Airline
avg_delay = df.groupby("AIRLINE")["ARRIVAL_DELAY"].mean().sort_values()

plt.figure(figsize=(10, 6))
avg_delay.plot(kind="bar")
plt.title("Average Arrival Delay by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Arrival Delay (minutes)")
plt.tight_layout()
plt.savefig("outputs/average_delay_by_airline.png")
plt.show()

# 3. Flights by Month
monthly_flights = df["MONTH"].value_counts().sort_index()

plt.figure(figsize=(10, 6))
monthly_flights.plot(kind="bar")
plt.title("Number of Flights by Month")
plt.xlabel("Month")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig("outputs/flights_by_month.png")
plt.show()

# 4. Arrival Delay Distribution
plt.figure(figsize=(10, 6))
df["ARRIVAL_DELAY"].dropna().clip(-50, 200).plot(kind="hist", bins=50)
plt.title("Arrival Delay Distribution")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig("outputs/arrival_delay_distribution.png")
plt.show()

print("\n========== EDA SUMMARY ==========")
print("\nFlights by airline:")
print(airline_counts)

print("\nAverage arrival delay:")
print(avg_delay)

print("\nEDA completed successfully!")