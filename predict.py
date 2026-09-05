import pandas as pd
import joblib

print("======================================")
print("     FLIGHT DELAY PREDICTION SYSTEM")
print("======================================")

# Load model
model = joblib.load(
    "models/flight_delay_model.pkl"
)

# Load frequency mappings
frequency_mappings = joblib.load(
    "models/frequency_mappings.pkl"
)

# Prediction threshold
THRESHOLD = 0.20

print("\nEnter flight details:\n")

# Get user input
month = int(input("Month (1-12): "))
day = int(input("Day (1-31): "))
day_of_week = int(input("Day of Week (1=Mon, 7=Sun): "))

airline = input("Airline code (e.g. AA, DL, UA): ").strip().upper()

origin = input(
    "Origin Airport (e.g. ATL): "
).strip().upper()

destination = input(
    "Destination Airport (e.g. LAX): "
).strip().upper()

scheduled_departure = int(
    input("Scheduled Departure (HHMM): ")
)

distance = float(
    input("Distance (miles): ")
)

scheduled_time = float(
    input("Scheduled Flight Time (minutes): ")
)

# --------------------------------------
# Feature engineering
# --------------------------------------

dep_hour = scheduled_departure // 100
dep_minute = scheduled_departure % 100

is_weekend = (
    1 if day_of_week >= 6 else 0
)

route = (
    origin + "_" + destination
)

airline_route = (
    airline + "_" + route
)

# Distance category
if distance < 500:
    distance_category = "Short"
elif distance < 1500:
    distance_category = "Medium"
elif distance < 2500:
    distance_category = "Long"
else:
    distance_category = "Very_Long"

# --------------------------------------
# Create input dataframe
# --------------------------------------

input_df = pd.DataFrame([{
    "MONTH": month,
    "DAY": day,
    "DAY_OF_WEEK": day_of_week,
    "SCHEDULED_DEPARTURE": scheduled_departure,
    "DEP_HOUR": dep_hour,
    "DEP_MINUTE": dep_minute,
    "DISTANCE": distance,
    "SCHEDULED_TIME": scheduled_time,
    "IS_WEEKEND": is_weekend,
    "AIRLINE": airline,
    "ORIGIN_AIRPORT": origin,
    "DESTINATION_AIRPORT": destination,
    "ROUTE": route,
    "AIRLINE_ROUTE": airline_route,
    "DISTANCE_CATEGORY": distance_category
}])

# --------------------------------------
# Frequency encoding
# --------------------------------------

categorical_columns = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "ROUTE",
    "AIRLINE_ROUTE",
    "DISTANCE_CATEGORY"
]

for col in categorical_columns:

    frequency = frequency_mappings[col]

    input_df[col + "_FREQ"] = (
        input_df[col]
        .map(frequency)
        .fillna(0)
    )

# --------------------------------------
# Final feature order
# --------------------------------------

numerical_columns = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_DEPARTURE",
    "DEP_HOUR",
    "DEP_MINUTE",
    "DISTANCE",
    "SCHEDULED_TIME",
    "IS_WEEKEND"
]

frequency_columns = [
    col + "_FREQ"
    for col in categorical_columns
]

final_features = (
    numerical_columns
    + frequency_columns
)

X = input_df[
    final_features
].astype(float)

# --------------------------------------
# Prediction
# --------------------------------------

probability = model.predict_proba(X)[0][1]

prediction = (
    "DELAYED"
    if probability >= THRESHOLD
    else "ON TIME"
)

# --------------------------------------
# Result
# --------------------------------------

print("\n======================================")
print("             PREDICTION")
print("======================================")

print(
    f"Delay Probability : {probability * 100:.2f}%"
)

print(
    f"Threshold          : {THRESHOLD * 100:.0f}%"
)

print(
    f"Prediction         : {prediction}"
)

print("======================================")

if prediction == "DELAYED":
    print("⚠️  Flight is likely to be delayed.")
else:
    print("✅ Flight is likely to be on time.")

