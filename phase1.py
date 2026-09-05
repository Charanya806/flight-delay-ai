import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============================================================
# STEP 1: GET THE DATA
# ============================================================
# For now we CREATE fake but realistic flight data so we can
# build and test the project immediately.
# Later, replace this with a real CSV file (from BTS website or Kaggle).

def create_sample_flight_data(number_of_flights=5000):
    np.random.seed(42)  # makes results repeatable every time we run it

    airlines = ["AA", "DL", "UA", "WN", "AS", "B6"]
    airports = ["ATL", "ORD", "DFW", "LAX", "JFK", "DEN", "SFO", "MIA"]

    data = pd.DataFrame({
        "airline": np.random.choice(airlines, number_of_flights),
        "origin_airport": np.random.choice(airports, number_of_flights),
        "destination_airport": np.random.choice(airports, number_of_flights),
        "departure_hour": np.random.randint(0, 24, number_of_flights),
        "temperature_F": np.random.normal(60, 20, number_of_flights).round(1),
        "wind_speed_mph": abs(np.random.normal(10, 6, number_of_flights)).round(1),
        "airport_traffic": np.random.randint(20, 300, number_of_flights),
    })

    # Simulate whether flight was delayed (1 = Yes, 0 = No)
    # Bad weather + heavy traffic + rush hour = higher delay chance
    delay_chance = (
        0.1
        + 0.15 * (data["wind_speed_mph"] > 20)
        + 0.15 * (data["airport_traffic"] > 200)
        + 0.1 * data["departure_hour"].isin([7, 8, 17, 18])
    )
    data["delayed"] = np.random.binomial(1, delay_chance.clip(0, 0.9))

    # Sprinkle in some missing values on purpose (real data always has these)
    for col in ["temperature_F", "wind_speed_mph", "airport_traffic"]:
        blank_rows = data.sample(frac=0.02).index
        data.loc[blank_rows, col] = np.nan

    return data


# ============================================================
# STEP 2: CLEAN THE DATA
# ============================================================
def clean_the_data(data):
    # Remove exact duplicate rows (like removing photocopies)
    data = data.drop_duplicates()

    # Fill missing numbers with the median (a safe "typical" value)
    number_columns = ["temperature_F", "wind_speed_mph", "airport_traffic"]
    for col in number_columns:
        data[col] = data[col].fillna(data[col].median())

    return data


# ============================================================
# STEP 3: CONVERT TEXT TO NUMBERS
# ============================================================
# Machine learning models can't read "AA" or "ORD" - they need numbers.
# LabelEncoder just assigns each unique text value a number.
# Example: AA -> 0, DL -> 1, UA -> 2 ...
def convert_text_to_numbers(data):
    text_columns = ["airline", "origin_airport", "destination_airport"]
    for col in text_columns:
        encoder = LabelEncoder()
        data[col + "_code"] = encoder.fit_transform(data[col])
    return data


# ============================================================
# RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    print("Step 1: Creating flight data...")
    flights = create_sample_flight_data()
    print(f"  -> {len(flights)} flight records created")

    print("Step 2: Cleaning the data...")
    flights = clean_the_data(flights)
    print(f"  -> {len(flights)} records after cleaning")

    print("Step 3: Converting airline/airport names to numbers...")
    flights = convert_text_to_numbers(flights)

    flights.to_csv("cleaned_flight_data.csv", index=False)
    print("\nDone! Saved as 'cleaned_flight_data.csv'")
    print("\nFirst 5 rows:")
    print(flights.head())