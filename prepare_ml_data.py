import pandas as pd
import os

print("Loading cleaned dataset...")

df = pd.read_csv(
    "data/processed/flights_cleaned.csv",
    low_memory=False
)

print("Original shape:", df.shape)

# -----------------------------------------
# REMOVE CANCELLED FLIGHTS
# -----------------------------------------

df = df[df["CANCELLED"] == 0]

# Arrival delay must exist
df = df.dropna(
    subset=["ARRIVAL_DELAY"]
)

# -----------------------------------------
# TARGET
# -----------------------------------------

df["DELAYED"] = (
    df["ARRIVAL_DELAY"] > 15
).astype(int)

# -----------------------------------------
# TIME FEATURES
# -----------------------------------------

df["DEP_HOUR"] = (
    df["SCHEDULED_DEPARTURE"] // 100
).astype(int)

df["DEP_MINUTE"] = (
    df["SCHEDULED_DEPARTURE"] % 100
).astype(int)

# -----------------------------------------
# WEEKEND
# -----------------------------------------

df["IS_WEEKEND"] = (
    df["DAY_OF_WEEK"] >= 6
).astype(int)

# -----------------------------------------
# ROUTE
# -----------------------------------------

df["ORIGIN_AIRPORT"] = (
    df["ORIGIN_AIRPORT"].astype(str)
)

df["DESTINATION_AIRPORT"] = (
    df["DESTINATION_AIRPORT"].astype(str)
)

df["AIRLINE"] = (
    df["AIRLINE"].astype(str)
)

df["ROUTE"] = (
    df["ORIGIN_AIRPORT"]
    + "_"
    + df["DESTINATION_AIRPORT"]
)

# -----------------------------------------
# AIRLINE + ROUTE
# -----------------------------------------

df["AIRLINE_ROUTE"] = (
    df["AIRLINE"]
    + "_"
    + df["ROUTE"]
)

# -----------------------------------------
# DISTANCE CATEGORY
# -----------------------------------------

def distance_category(distance):

    if distance < 500:
        return "Short"

    elif distance < 1500:
        return "Medium"

    elif distance < 2500:
        return "Long"

    else:
        return "Very_Long"


df["DISTANCE_CATEGORY"] = (
    df["DISTANCE"].apply(distance_category)
)

# -----------------------------------------
# SCHEDULED TIME
# -----------------------------------------

df["SCHEDULED_TIME"] = pd.to_numeric(
    df["SCHEDULED_TIME"],
    errors="coerce"
)

# -----------------------------------------
# SELECT FEATURES
# -----------------------------------------

features = [

    "MONTH",
    "DAY",
    "DAY_OF_WEEK",

    "AIRLINE",

    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",

    "SCHEDULED_DEPARTURE",
    "DEP_HOUR",
    "DEP_MINUTE",

    "DISTANCE",
    "SCHEDULED_TIME",

    "IS_WEEKEND",

    "ROUTE",
    "AIRLINE_ROUTE",

    "DISTANCE_CATEGORY"
]

target = "DELAYED"

ml_df = df[
    features + [target]
].copy()

# -----------------------------------------
# SAMPLE DATA
# -----------------------------------------

if len(ml_df) > 500000:

    ml_df = ml_df.sample(
        n=500000,
        random_state=42
    )

# -----------------------------------------
# SAVE
# -----------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = (
    "data/processed/ml_dataset.csv"
)

ml_df.to_csv(
    output_path,
    index=False
)

# -----------------------------------------
# OUTPUT
# -----------------------------------------

print("\n========== ML DATASET ==========")

print(
    "Shape:",
    ml_df.shape
)

print("\nFeatures:")

for feature in features:
    print("-", feature)

print("\nTarget distribution:")

print(
    ml_df["DELAYED"].value_counts()
)

print("\nSaved to:")

print(output_path)

print(
    "\nML dataset preparation completed successfully!"
)

