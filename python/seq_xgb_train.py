#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ----------------------------------------------------
# 1. Define the sequential ensemble classifier
# ----------------------------------------------------
class SequentialEnsembleClassifier:
    """
    Uses a two-stage approach:
      - stage1 (threshold) for gating into the 'high-risk' subset
      - stage2 for re-classifying only high-risk samples
      - final classification uses threshold=0.5 on stage2's probabilities
    """
    def __init__(self, stage1, threshold, stage2):
        self.stage1 = stage1
        self.threshold = threshold  # Stage 1 gating threshold
        self.stage2 = stage2
        
    def predict_proba(self, X, **kwargs):
        # Probability from Stage 1
        proba1 = self.stage1.predict_proba(X, **kwargs)[:, 1]
        final_proba = np.zeros(X.shape[0], dtype=float)
        
        # Identify high-risk samples
        high_risk_indices = np.where(proba1 >= self.threshold)[0]
        if len(high_risk_indices) > 0:
            if isinstance(X, pd.DataFrame):
                X_high = X.iloc[high_risk_indices]
            else:
                X_high = X[high_risk_indices]
            
            proba2 = self.stage2.predict_proba(X_high, **kwargs)[:, 1]
            final_proba[high_risk_indices] = proba2
        
        # Return two-class probabilities
        return np.column_stack((1 - final_proba, final_proba))
    
    def predict(self, X, **kwargs):
        # Stage 2 threshold = 0.5
        proba = self.predict_proba(X, **kwargs)[:, 1]
        return (proba >= 0.5).astype(int)

# ----------------------------------------------------
# 2. Helper to find threshold for a target TPR
# ----------------------------------------------------
def find_threshold_for_target_recall(y_true, y_proba, target_tpr=0.90):
    """
    Find the smallest threshold for y_proba at which TPR >= target_tpr.
    If no threshold achieves TPR >= target_tpr, return the lowest threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx_candidates = np.where(tpr >= target_tpr)[0]
    if len(idx_candidates) == 0:
        # If no threshold meets the TPR requirement, pick the last threshold
        return thresholds[-1]
    else:
        # The first threshold where TPR crosses target_tpr
        first_idx = idx_candidates[0]
        return thresholds[first_idx]

def main():
    # ----------------------------------------------------
    # 3. Load the data
    # ----------------------------------------------------
    data_dir = "/well/clifton/users/ncu080/ad4mm/data"
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    
    print("Loading training data...")
    X_train = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)
    
    if y_train_df.shape[1] == 1:
        y_train = y_train_df.iloc[:, 0]
    else:
        y_train = y_train_df.values.ravel()
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    # ----------------------------------------------------
    # 4. Stage 1: Train the first model on the entire dataset
    # ----------------------------------------------------
    model_stage1 = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',  # CPU
        eval_metric='auc'
    )
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_stage1 = GridSearchCV(
        estimator=model_stage1,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    print("Starting grid search for Stage 1...")
    grid_search_stage1.fit(X_train, y_train)
    print("\nGrid search for Stage 1 complete.")
    print("Best parameters for Stage 1:", grid_search_stage1.best_params_)
    print("Best cross-validation ROC AUC for Stage 1:", grid_search_stage1.best_score_)
    
    best_model_stage1 = grid_search_stage1.best_estimator_
    
    # ----------------------------------------------------
    # 5. Choose a Stage 1 threshold by picking threshold for target TPR
    # ----------------------------------------------------
    y_train_proba_stage1 = best_model_stage1.predict_proba(X_train)[:, 1]
    
    # Let's choose TPR = 0.50 for the high-risk gating
    target_tpr = 0.50
    threshold_stage1 = find_threshold_for_target_recall(y_train, y_train_proba_stage1, target_tpr=target_tpr)
    
    print(f"Selected Stage 1 threshold for TPR>={target_tpr}: {threshold_stage1:.6f}")
    
    high_risk_indices = np.where(y_train_proba_stage1 >= threshold_stage1)[0]
    print(f"Number of samples flagged as high risk by Stage 1: {len(high_risk_indices)} / {len(y_train)}")
    
    X_train_high = X_train.iloc[high_risk_indices]
    y_train_high = y_train.iloc[high_risk_indices]
    
    # ----------------------------------------------------
    # 6. Stage 2: Train the second model on the high-risk subset
    # ----------------------------------------------------
    model_stage2 = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='auc'
    )
    
    grid_search_stage2 = GridSearchCV(
        estimator=model_stage2,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    print("Starting grid search for Stage 2 on high-risk subset...")
    grid_search_stage2.fit(X_train_high, y_train_high)
    print("\nGrid search for Stage 2 complete.")
    print("Best parameters for Stage 2:", grid_search_stage2.best_params_)
    print("Best cross-validation ROC AUC for Stage 2:", grid_search_stage2.best_score_)
    
    best_model_stage2 = grid_search_stage2.best_estimator_
    
    # ----------------------------------------------------
    # 7. Create the final two-stage ensemble
    # ----------------------------------------------------
    ensemble_model = SequentialEnsembleClassifier(
        stage1=best_model_stage1,
        threshold=threshold_stage1,
        stage2=best_model_stage2
    )
    
    # ----------------------------------------------------
    # 8. Evaluate on the training data (Stage 1 threshold + Stage 2 threshold=0.5)
    # ----------------------------------------------------
    final_pred = ensemble_model.predict(X_train)
    final_proba = ensemble_model.predict_proba(X_train)[:, 1]
    
    cm = confusion_matrix(y_train, final_pred)
    report = classification_report(y_train, final_pred, digits=4)
    auc_score = roc_auc_score(y_train, final_proba)
    
    performance_metrics = f"""
Sequential Ensemble Performance Metrics (Less Stringent):
----------------------------------------------------------
Stage 1 Threshold (target TPR >= {target_tpr}): {threshold_stage1:.6f}
Confusion Matrix:
{cm}

Classification Report:
{report}

ROC AUC Score: {auc_score:.4f}
"""
    print(performance_metrics)
    
    # ----------------------------------------------------
    # 9. Save performance metrics & the model
    # ----------------------------------------------------
    metrics_dir = "/well/clifton/users/ncu080/ad4mm/outputs/performace_metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "sequential_ensemble_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(performance_metrics)
    print(f"Performance metrics saved to: {metrics_path}")
    
    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "sequential_ensemble_model.pkl")
    joblib.dump(ensemble_model, model_save_path)
    print(f"Sequential ensemble model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
