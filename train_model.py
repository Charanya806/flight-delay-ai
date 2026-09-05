import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
import joblib
import os

print("Loading ML dataset...")

df = pd.read_csv(
    "data/processed/ml_dataset.csv",
    low_memory=False
)

print("Dataset shape:", df.shape)

# -----------------------------------------
# SPLIT FEATURES AND TARGET
# -----------------------------------------

X = df.drop("DELAYED", axis=1)
y = df["DELAYED"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------------
# CATEGORICAL FEATURES
# -----------------------------------------

categorical_columns = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "ROUTE",
    "AIRLINE_ROUTE",
    "DISTANCE_CATEGORY"
]

# -----------------------------------------
# FREQUENCY ENCODING
# -----------------------------------------

print("\nEncoding categorical features...")

frequency_mappings = {}

for col in categorical_columns:

    frequency = (
        X_train[col]
        .value_counts(normalize=True)
    )

    frequency_mappings[col] = frequency.to_dict()

    X_train[col + "_FREQ"] = (
        X_train[col]
        .map(frequency)
        .fillna(0)
    )

    X_test[col + "_FREQ"] = (
        X_test[col]
        .map(frequency)
        .fillna(0)
    )

# -----------------------------------------
# NUMERICAL FEATURES
# -----------------------------------------

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

X_train_final = X_train[
    final_features
].astype(float)

X_test_final = X_test[
    final_features
].astype(float)

print(
    "Final training shape:",
    X_train_final.shape
)

print(
    "Final testing shape:",
    X_test_final.shape
)

# -----------------------------------------
# HISTOGRAM GRADIENT BOOSTING
# -----------------------------------------

print("\nTraining Gradient Boosting model...")

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42
)

model.fit(
    X_train_final,
    y_train
)

# -----------------------------------------
# PREDICTION
# -----------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test_final
)

y_probability = model.predict_proba(
    X_test_final
)[:, 1]

# -----------------------------------------
# EVALUATION
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== MODEL RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# -----------------------------------------
# SAVE MODEL
# -----------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/flight_delay_model.pkl"
)

joblib.dump(
    final_features,
    "models/model_features.pkl"
)

joblib.dump(
    frequency_mappings,
    "models/frequency_mappings.pkl"
)

print("\n===================================")
print("Model saved successfully!")
print("models/flight_delay_model.pkl")
print("models/model_features.pkl")
print("models/frequency_mappings.pkl")
print("===================================")

