#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import xgboost as xgb

# ===============================
# Define the Sequential Ensemble Classifier
# ===============================
class SequentialEnsembleClassifier:
    def __init__(self, stage1, threshold, stage2):
        self.stage1 = stage1
        self.threshold = threshold  # Stage 1 gating threshold
        self.stage2 = stage2
        
    def predict_proba(self, X, **kwargs):
        proba1 = self.stage1.predict_proba(X, **kwargs)[:, 1]
        final_proba = np.zeros(X.shape[0], dtype=float)
        high_risk_indices = np.where(proba1 >= self.threshold)[0]
        
        if len(high_risk_indices) > 0:
            if isinstance(X, pd.DataFrame):
                X_high = X.iloc[high_risk_indices]
            else:
                X_high = X[high_risk_indices]
            proba2 = self.stage2.predict_proba(X_high, **kwargs)[:, 1]
            final_proba[high_risk_indices] = proba2
        
        return np.column_stack((1 - final_proba, final_proba))
    
    def predict(self, X, **kwargs):
        # Stage 2 threshold = 0.5
        proba = self.predict_proba(X, **kwargs)[:, 1]
        return (proba >= 0.5).astype(int)

# ===============================
# Optional: Tee class to log output
# ===============================
class Tee:
    """
    Duplicate the output to multiple streams (console and log file).
    """
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

def compute_extended_metrics(y_true, y_pred, y_proba):
    """
    Computes various classification metrics including Sensitivity, Specificity, PPV, and NPV.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / TPR
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    auc_score = roc_auc_score(y_true, y_proba)
    
    sensitivity = recall  # True Positive Rate (TPR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (TNR)
    ppv = precision  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_score,
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv
    }

def format_metrics(metrics, dataset_name="Dataset"):
    """
    Formats metrics into a printable string.
    """
    return f"""{dataset_name} Metrics:
  Precision: {metrics["precision"]:.4f}
  Recall (Sensitivity): {metrics["recall"]:.4f}
  F1 Score: {metrics["f1_score"]:.4f}
  AUC-ROC: {metrics["auc_roc"]:.4f}
  Confusion Matrix:
{metrics["confusion_matrix"]}
  Sensitivity (TPR): {metrics["sensitivity"]:.4f}
  Specificity (TNR): {metrics["specificity"]:.4f}
  PPV (Positive Predictive Value): {metrics["ppv"]:.4f}
  NPV (Negative Predictive Value): {metrics["npv"]:.4f}
"""

def evaluate_ensemble(ensemble_model, X, y, dataset_name="Data"):
    """
    Evaluates the given ensemble model on data (X,y).
    Prints confusion matrix, classification report, and AUC.
    Returns a string with formatted performance metrics.
    """
    pred = ensemble_model.predict(X)
    proba = ensemble_model.predict_proba(X)[:, 1]
    
    metrics = compute_extended_metrics(y, pred, proba)
    
    return format_metrics(metrics, dataset_name)

def main():
    DATA_DIR = "/well/clifton/users/ncu080/ad4mm/data"
    MODEL_DIR = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    OUTPUT_DIR = "/well/clifton/users/ncu080/ad4mm/outputs/performace_metrics"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DIR, "seq_evaluation.txt")

    # If you want both console + file logging, use the Tee mechanism
    original_stdout = sys.stdout
    try:
        with open(output_file_path, "w") as f:
            # Toggle the line below if you only want file or both console+file
            # sys.stdout = Tee(original_stdout, f)
            sys.stdout = f  # only write to file

            print("Loading training data...")
            X_train = pd.read_csv(os.path.join(DATA_DIR, "x_train.csv"))
            y_train_df = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
            if y_train_df.shape[1] == 1:
                y_train = y_train_df.iloc[:, 0]
            else:
                y_train = y_train_df.values.ravel()

            print("Loading test data...")
            X_test = pd.read_csv(os.path.join(DATA_DIR, "x_test.csv"))
            y_test_df = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
            if y_test_df.shape[1] == 1:
                y_test = y_test_df.iloc[:, 0]
            else:
                y_test = y_test_df.values.ravel()

            print("\n--- Loading Sequential Ensemble Model ---")
            model_path = os.path.join(MODEL_DIR, "sequential_ensemble_model.pkl")
            ensemble_model = joblib.load(model_path)

            print("\n--- Evaluating on Training Data ---")
            train_metrics = evaluate_ensemble(ensemble_model, X_train, y_train, dataset_name="Training")
            print(train_metrics)

            print("\n--- Evaluating on Test Data ---")
            test_metrics = evaluate_ensemble(ensemble_model, X_test, y_test, dataset_name="Test")
            print(test_metrics)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()
