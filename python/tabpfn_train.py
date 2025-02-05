#Pre-process data for TabPFN model

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load data
data_dir = "/well/clifton/users/ncu080/ad4mm/data"
x_train_path = f"{data_dir}/x_train.csv"
y_train_path = f"{data_dir}/y_train.csv"

X_train = pd.read_csv(x_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()  # Ensure it's a 1D array

# Standardize features (TabPFN expects normalized data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler for test inference
from joblib import dump
dump(scaler, "/well/clifton/users/ncu080/ad4mm/outputs/models/tabpfn_scaler.pkl")

print("Data successfully preprocessed.")

# Train TabPFN model

from tabpfn import TabPFNClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Instantiate TabPFN model
model = TabPFNClassifier(device="cuda")

# Fit TabPFN to the training data
model.fit(X_train_scaled, y_train)

# Predict probabilities (use only the positive class probability as an anomaly score)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]

# Convert probabilities into binary classification
threshold = np.percentile(y_train_proba, 95)  # Define a threshold (e.g., top 5% as anomalies)
y_train_pred = (y_train_proba > threshold).astype(int)

# Evaluate performance on the training set
auc_score = roc_auc_score(y_train, y_train_proba)
print(f"Training ROC AUC: {auc_score:.3f}")

print("\nClassification Report:")
print(classification_report(y_train, y_train_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

# Save trained model
import os
from joblib import dump

model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
os.makedirs(model_dir, exist_ok=True)
dump(model, os.path.join(model_dir, "tabpfn_model.joblib"))

print(f"Model saved to: {model_dir}/tabpfn_model.joblib")
