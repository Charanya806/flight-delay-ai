import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

print("Loading dataset and model...")

# Load dataset
df = pd.read_csv(
    "data/processed/ml_dataset.csv",
    low_memory=False
)

# Load trained model
model = joblib.load(
    "models/flight_delay_model.pkl"
)

# Load frequency mappings
frequency_mappings = joblib.load(
    "models/frequency_mappings.pkl"
)

# Decision threshold
THRESHOLD = 0.20

X = df.drop(
    "DELAYED",
    axis=1
)

y = df["DELAYED"]

# Same train/test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Test samples:", len(X_test))

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
# PREDICTION
# =========================================================

print("Generating predictions...")

y_probability = (
    model.predict_proba(
        X_test_final
    )[:, 1]
)

# Apply selected threshold
y_pred = (
    y_probability >= THRESHOLD
).astype(int)

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("       MODEL EVALUATION RESULTS")
print("======================================")

print(
    f"\nDecision Threshold : {THRESHOLD * 100:.0f}%"
)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "On Time",
            "Delayed"
        ]
    )
)

# Create outputs folder
os.makedirs(
    "outputs",
    exist_ok=True
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "On Time",
        "Delayed"
    ]
)

disp.plot(
    values_format="d"
)

plt.title(
    "Flight Delay Prediction - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "\nSaved: outputs/confusion_matrix.png"
)

# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Flight Delay Prediction"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png",
    dpi=300
)

plt.close()

print(
    "Saved: outputs/roc_curve.png"
)

# =========================================================
# PRECISION-RECALL CURVE
# =========================================================

precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Flight Delay Prediction"
)

plt.tight_layout()

plt.savefig(
    "outputs/precision_recall_curve.png",
    dpi=300
)

plt.close()

print(
    "Saved: outputs/precision_recall_curve.png"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n======================================")
print("Evaluation completed successfully!")
print("======================================")

print("\nGenerated files:")

print(
    "1. outputs/confusion_matrix.png"
)

print(
    "2. outputs/roc_curve.png"
)

print(
    "3. outputs/precision_recall_curve.png"
)

print("\nModel ROC-AUC:", round(roc_auc, 4))

print(
    "\nNOTE:"
)

print(
    "Accuracy should not be considered alone "
    "because the dataset is imbalanced."
)

print(
    "The 20% threshold is used to improve "
    "detection of delayed flights."
)

