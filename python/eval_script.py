#!/usr/bin/env python3
"""
Master evaluation script:
  - Reads in training and test data.
  - Loads the trained models.
  - For each model, computes evaluation metrics (precision, recall, F1 score, AUC-ROC, etc.)
  - Uses a training-derived threshold (via Youden's index) for test data evaluation.
  
Models evaluated include:
    - Autoencoder (PyTorch)
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - Logistic Regression with SMOTE
    - Logistic Regression (plain)
    - One-Class SVM
    - TabPFN
    - XGBoost with SMOTE
    - XGBoost (regular)
    - Sequential Ensemble
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

#####################################
# Helper class to duplicate output (Tee)
#####################################
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

#####################################
# Helper functions for metric computation
#####################################
def compute_metrics(y_true, scores, threshold=None):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    if threshold is None:
        threshold = optimal_threshold

    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    auc = roc_auc_score(y_true, scores)

    return {
        "optimal_threshold": optimal_threshold,
        "used_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": precision,
        "NPV": npv
    }

def print_metrics(metrics):
    print(f"  Optimal Threshold (from training): {metrics['optimal_threshold']:.4f}")
    print(f"  Used Threshold: {metrics['used_threshold']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")
    print("  Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  PPV: {metrics['PPV']:.4f}")
    print(f"  NPV: {metrics['NPV']:.4f}")

def evaluate_model(model_name, y_train, y_test, train_scores, test_scores):
    print(f"\n=== {model_name} Evaluation ===")
    print("\nTraining Data Metrics:")
    metrics_train = compute_metrics(y_train, train_scores)
    print_metrics(metrics_train)
    threshold = metrics_train["optimal_threshold"]
    print("\nTest Data Metrics (using training threshold):")
    metrics_test = compute_metrics(y_test, test_scores, threshold=threshold)
    print_metrics(metrics_test)
    return metrics_train, metrics_test

def evaluate_autoencoder(X_train_raw, X_test_raw, y_train, y_test, model_dir):
    print("\n--- Autoencoder ---")
    scaler_path = os.path.join(model_dir, "autoencoder_scaler.pkl")
    model_path = os.path.join(model_dir, "autoencoder_model.pth")
    scaler = joblib.load(scaler_path)
    X_train = scaler.transform(X_train_raw.values)
    X_test = scaler.transform(X_test_raw.values)
    input_dim = X_train.shape[1]
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=4):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 15),
                nn.ReLU(),
                nn.Linear(15, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 15),
                nn.ReLU(),
                nn.Linear(15, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).to(device)
        recon_train = model(X_train_tensor)
        recon_test  = model(X_test_tensor)
        train_errors = torch.mean((recon_train - X_train_tensor)**2, dim=1).cpu().numpy()
        test_errors  = torch.mean((recon_test - X_test_tensor)**2, dim=1).cpu().numpy()
    evaluate_model("Autoencoder", y_train, y_test, train_errors, test_errors)

def evaluate_isolation_forest(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- Isolation Forest ---")
    model_path = os.path.join(model_dir, "best_isolation_forest.joblib")
    model = joblib.load(model_path)
    train_scores = - model.decision_function(X_train)
    test_scores  = - model.decision_function(X_test)
    evaluate_model("Isolation Forest", y_train, y_test, train_scores, test_scores)

def evaluate_lof(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- Local Outlier Factor ---")
    model_path = os.path.join(model_dir, "lof_novelty_best_model.joblib")
    model = joblib.load(model_path)
    train_scores = - model.score_samples(X_train)
    test_scores  = - model.score_samples(X_test)
    evaluate_model("Local Outlier Factor", y_train, y_test, train_scores, test_scores)

def evaluate_logistic_smote(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- Logistic Regression (SMOTE) ---")
    model_path = os.path.join(model_dir, "logistic_smote_best_model.pkl")
    model = joblib.load(model_path)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores  = model.predict_proba(X_test)[:, 1]
    evaluate_model("Logistic Regression (SMOTE)", y_train, y_test, train_scores, test_scores)

def evaluate_logistic_regression(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- Logistic Regression ---")
    model_path = os.path.join(model_dir, "logistic_regression_best_model.pkl")
    model = joblib.load(model_path)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores  = model.predict_proba(X_test)[:, 1]
    evaluate_model("Logistic Regression", y_train, y_test, train_scores, test_scores)

def evaluate_one_class_svm(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- One-Class SVM ---")
    model_path = os.path.join(model_dir, "best_one_class_svm.joblib")
    model = joblib.load(model_path)
    train_scores = - model.decision_function(X_train)
    test_scores  = - model.decision_function(X_test)
    evaluate_model("One-Class SVM", y_train, y_test, train_scores, test_scores)

def evaluate_tabpfn(X_train_raw, X_test_raw, y_train, y_test, model_dir):
    if TabPFNClassifier is None:
        print("\n--- TabPFN not evaluated because 'tabpfn' is not installed.")
        return
    print("\n--- TabPFN ---")
    model_path = os.path.join(model_dir, "tabpfn_model.joblib")
    scaler_path = os.path.join(model_dir, "tabpfn_scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_train = scaler.transform(X_train_raw.values)
    X_test  = scaler.transform(X_test_raw.values)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores  = model.predict_proba(X_test)[:, 1]
    evaluate_model("TabPFN", y_train, y_test, train_scores, test_scores)

def evaluate_xgboost_smote(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- XGBoost with SMOTE ---")
    model_path = os.path.join(model_dir, "best_xgboost_smote_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores  = model.predict_proba(X_test)[:, 1]
    evaluate_model("XGBoost with SMOTE", y_train, y_test, train_scores, test_scores)

def evaluate_xgboost(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- XGBoost ---")
    model_path = os.path.join(model_dir, "best_xgboost_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores  = model.predict_proba(X_test)[:, 1]
    evaluate_model("XGBoost", y_train, y_test, train_scores, test_scores)

# New evaluation function for the Sequential Ensemble model
def evaluate_sequential_ensemble(X_train, X_test, y_train, y_test, model_dir):
    print("\n--- Sequential Ensemble ---")
    model_path = os.path.join(model_dir, "sequential_ensemble_model.pkl")
    ensemble_model = joblib.load(model_path)
    train_scores = ensemble_model.predict_proba(X_train)[:, 1]
    test_scores  = ensemble_model.predict_proba(X_test)[:, 1]
    evaluate_model("Sequential Ensemble", y_train, y_test, train_scores, test_scores)

def main():
    DATA_DIR = "/well/clifton/users/ncu080/ad4mm/data"
    MODEL_DIR = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    OUTPUT_DIR = "/well/clifton/users/ncu080/ad4mm/outputs/performace_metrics"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
    
    original_stdout = sys.stdout
    with open(output_file_path, "w") as output_file:
        sys.stdout = Tee(original_stdout, output_file)
        
        print("Loading training data...")
        x_train_path = os.path.join(DATA_DIR, "x_train.csv")
        y_train_path = os.path.join(DATA_DIR, "y_train.csv")
        x_test_path  = os.path.join(DATA_DIR, "x_test.csv")
        y_test_path  = os.path.join(DATA_DIR, "y_test.csv")
        
        X_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path)
        y_train = y_train.iloc[:, 0].values if y_train.shape[1] == 1 else y_train.values.ravel()
        
        print("Loading test data...")
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path)
        y_test = y_test.iloc[:, 0].values if y_test.shape[1] == 1 else y_test.values.ravel()
        
        evaluate_autoencoder(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_isolation_forest(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_lof(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_logistic_smote(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_logistic_regression(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_one_class_svm(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_tabpfn(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_xgboost_smote(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_xgboost(X_train, X_test, y_train, y_test, MODEL_DIR)
        evaluate_sequential_ensemble(X_train, X_test, y_train, y_test, MODEL_DIR)
        
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()
