from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load data
data_dir = "/well/clifton/users/ncu080/ad4mm/data"
x_train_path = f"{data_dir}/x_train.csv"
y_train_path = f"{data_dir}/y_train.csv"

X_train = pd.read_csv(x_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()  # Ensure it's a 1D array

# Subset data: Keep all y = 1 and randomly sample 9850 instances from y = 0
positive_indices = np.where(y_train == 1)[0]  # Indices where y = 1
negative_indices = np.where(y_train == 0)[0]  # Indices where y = 0

# Randomly sample 9850 negative instances
np.random.seed(42)  # For reproducibility
sampled_negative_indices = np.random.choice(negative_indices, size=9850, replace=False)

# Combine selected indices
selected_indices = np.concatenate([positive_indices, sampled_negative_indices])

# Subset the data
X_train_subset = X_train.iloc[selected_indices]
y_train_subset = y_train[selected_indices]

# Standardize features (TabPFN expects normalized data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_subset)

# Save the scaler for test inference
from joblib import dump
dump(scaler, "/well/clifton/users/ncu080/ad4mm/outputs/models/tabpfn_scaler.pkl")

print(f"Data successfully preprocessed. Subset size: {X_train_scaled.shape}")

# Train TabPFN model
from tabpfn import TabPFNClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Instantiate TabPFN model
model = TabPFNClassifier(device="cuda")

# Fit TabPFN to the subset training data
model.fit(X_train_scaled, y_train_subset)

# Predict probabilities (use only the positive class probability as an anomaly score)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]

# Convert probabilities into binary classification
threshold = np.percentile(y_train_proba, 99.7)  # Define a 0.3% threshold based on prior knowledge
y_train_pred = (y_train_proba > threshold).astype(int)

# Evaluate performance on the training set
auc_score = roc_auc_score(y_train_subset, y_train_proba)
print(f"Training ROC AUC: {auc_score:.3f}")

print("\nClassification Report:")
print(classification_report(y_train_subset, y_train_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_train_subset, y_train_pred))

# Save trained model
import os
from joblib import dump

model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
os.makedirs(model_dir, exist_ok=True)
dump(model, os.path.join(model_dir, "tabpfn_model.joblib"))

print(f"Model saved to: {model_dir}/tabpfn_model.joblib")
