#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import dump

def iforest_auc_score(estimator, X, y):
    """
    Custom scoring function for IsolationForest.
    We invert decision_function so higher = outlier.
    Then compute roc_auc_score with y (where y=1 => outlier).
    """
    decision_scores = estimator.decision_function(X)  # Higher => inlier
    outlier_scores = -decision_scores                # Flip => Higher => outlier
    return roc_auc_score(y, outlier_scores)

def main():
    data_dir = "/well/clifton/users/ncu080/ad4mm/data"
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    print("Loading training data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Unique labels in y_train:", y_train.unique())

    iforest_base = IsolationForest(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_samples': [0.5, 1.0],
        'contamination': [0.003],
        'max_features': [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pass the iforest_auc_score directly to scoring:
    grid_search = GridSearchCV(
        estimator=iforest_base,
        param_grid=param_grid,
        scoring=iforest_auc_score,  # No make_scorer
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting Grid Search for IsolationForest...")
    grid_search.fit(X_train, y_train)
    print("Grid Search complete.")
    print("Best parameters found:", grid_search.best_params_)
    print("Best CV AUC:", grid_search.best_score_)

    best_iforest = grid_search.best_estimator_
    y_pred_if = best_iforest.predict(X_train)
    y_pred_mapped = [1 if p == -1 else 0 for p in y_pred_if]
    outlier_scores = -best_iforest.decision_function(X_train)

    if set(y_train.unique()) <= {0, 1}:
        print("\nClassification Report (treating y=1 as anomaly):")
        print(classification_report(y_train, y_pred_mapped, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_train, y_pred_mapped))

        auc_value = roc_auc_score(y_train, outlier_scores)
        print(f"Training ROC AUC with best model: {auc_value:.3f}")

    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_isolation_forest.joblib")
    dump(best_iforest, model_path)
    print(f"Best IsolationForest model saved to: {model_path}")

if __name__ == "__main__":
    main()
