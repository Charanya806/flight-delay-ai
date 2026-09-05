import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

print("Loading dataset and model...")

# Load dataset
df = pd.read_csv(
    "data/processed/ml_dataset.csv",
    low_memory=False
)

# Load model
model = joblib.load(
    "models/flight_delay_model.pkl"
)

# Load frequency mappings
frequency_mappings = joblib.load(
    "models/frequency_mappings.pkl"
)

X = df.drop(
    "DELAYED",
    axis=1
)

y = df["DELAYED"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# FREQUENCY ENCODING
# =========================================================

categorical_columns = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "ROUTE",
    "AIRLINE_ROUTE",
    "DISTANCE_CATEGORY"
]

print("Applying frequency encoding...")

for col in categorical_columns:

    frequency = frequency_mappings[col]

    X_test[
        col + "_FREQ"
    ] = (
        X_test[col]
        .map(frequency)
        .fillna(0)
    )

# =========================================================
# FINAL FEATURES
# =========================================================

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

# =========================================================
# SAMPLE TEST DATA
# =========================================================

# Permutation importance can be computationally expensive.
# Use a sample of 10,000 records for faster execution.

sample_size = min(
    10000,
    len(X_test_final)
)

sample_indices = np.random.RandomState(
    42
).choice(
    len(X_test_final),
    size=sample_size,
    replace=False
)

X_sample = X_test_final.iloc[
    sample_indices
]

y_sample = y_test.iloc[
    sample_indices
]

print(
    f"Calculating feature importance "
    f"using {sample_size} test samples..."
)

# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================

result = permutation_importance(
    model,
    X_sample,
    y_sample,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": final_features,
    "Importance": result.importances_mean
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("\n======================================")
print("       FEATURE IMPORTANCE")
print("======================================")

print(
    importance_df.to_string(
        index=False
    )
)

# =========================================================
# SAVE CSV
# =========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print(
    "\nSaved: outputs/feature_importance.csv"
)

# =========================================================
# TOP 10 FEATURES
# =========================================================

top_features = (
    importance_df
    .head(10)
    .sort_values(
        "Importance",
        ascending=True
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Permutation Importance (ROC-AUC)"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Features Influencing Flight Delay Prediction"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()

print(
    "Saved: outputs/feature_importance.png"
)

# =========================================================
# FINAL
# =========================================================

print("\n======================================")
print("Feature importance completed!")
print("======================================")

print(
    "\nTop 5 Important Features:"
)

for i, row in (
    importance_df.head(5)
    .reset_index(drop=True)
    .iterrows()
):
    print(
        f"{i + 1}. "
        f"{row['Feature']} "
        f"({row['Importance']:.6f})"
    )

