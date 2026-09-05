import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading dataset and model...")

df = pd.read_csv(
    "data/processed/ml_dataset.csv",
    low_memory=False
)

model = joblib.load(
    "models/flight_delay_model.pkl"
)

frequency_mappings = joblib.load(
    "models/frequency_mappings.pkl"
)

X = df.drop("DELAYED", axis=1)
y = df["DELAYED"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

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

    X_test[col + "_FREQ"] = (
        X_test[col]
        .map(frequency)
        .fillna(0)
    )

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

X_test_final = X_test[
    final_features
].astype(float)

print("Generating probabilities...")

probabilities = model.predict_proba(
    X_test_final
)[:, 1]

print("\n========== THRESHOLD RESULTS ==========")

thresholds = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10
]

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy*100:.2f}% | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n======================================")
print("BEST THRESHOLD:", best_threshold)
print("BEST F1 SCORE:", round(best_f1, 4))
print("======================================")
