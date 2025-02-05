#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from joblib import dump

def lof_auc_score(estimator, X, y):
    """
    Custom scorer for LOF using ROC AUC. We need to:
    1) Get 'score_samples' => higher = more inlier-like
    2) Flip sign so that higher = more outlier-like
    3) Compute roc_auc_score against y (where y=1 is outlier)
    """
    raw_scores = estimator.score_samples(X)   # higher => inlier
    outlier_scores = -raw_scores             # flip sign => higher => outlier
    return roc_auc_score(y, outlier_scores)

def main():
    # 1) Paths to your TRAIN data
    data_dir = "/well/clifton/users/ncu080/ad4mm/data"
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    # 2) Load training data
    print("Loading training data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    # Convert y_train to Series if it has only one column
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Unique labels in y_train:", y_train.unique())

    # 3) Base LOF with novelty=True so we can predict on new data later
    lof_base = LocalOutlierFactor(novelty=True)

    # 4) Define parameter grid
    # Keep contamination fixed at 0.003 (~0.3% outliers)
    # Example tuning for neighbors, distance metric, and leaf_size
    param_grid = {
        'n_neighbors': [5, 10, 20],
        'metric': ['euclidean', 'manhattan'],
        'leaf_size': [30, 60],
        'contamination': [0.003],  # FIXED
        'novelty': [True],         # Must remain True for out-of-sample usage
    }

    # 6) Stratified CV over the labeled data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 7) GridSearchCV
    print("\nStarting Grid Search...")
    grid_search = GridSearchCV(
        estimator=lof_base,
        param_grid=param_grid,
        scoring=lof_auc_score,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("Grid Search complete.")

    print("Best parameters found:", grid_search.best_params_)
    print("Best CV AUC:", grid_search.best_score_)

    # 8) Retrieve best estimator
    best_lof = grid_search.best_estimator_

    print("\nRefitting the best LOF on the entire training set...")
    # best_lof is already fitted by GridSearch on the entire CV, but we can refit manually if desired:
    # best_lof.fit(X_train)  # Not strictly necessary if 'refit=True' was set on GridSearchCV (default True)
    # We'll rely on the final fit from GridSearchCV which includes a fit on the entire dataset by default.

    # 9) Evaluate on TRAIN data
    y_pred_lof = best_lof.predict(X_train)  # +1 => inlier, -1 => outlier
    y_pred_mapped = [1 if val == -1 else 0 for val in y_pred_lof]

    lof_raw_scores = best_lof.score_samples(X_train)  # higher => inlier
    outlier_scores = -lof_raw_scores                  # flip => higher => outlier

    if set(y_train.unique()) <= {0, 1}:
        print("\n=== Classification Report (TRAIN) ===")
        print(classification_report(y_train, y_pred_mapped, zero_division=0))

        print("=== Confusion Matrix (TRAIN) ===")
        print(confusion_matrix(y_train, y_pred_mapped))

        train_auc = roc_auc_score(y_train, outlier_scores)
        print(f"TRAIN ROC AUC with best LOF: {train_auc:.3f}")
    else:
        print("\nWarning: y_train is not strictly {0,1}. Skipping classification metrics.")

    # 10) Save best LOF model for future usage on test data
    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lof_novelty_best_model.joblib")

    dump(best_lof, model_path)
    print(f"\nBest Local Outlier Factor (novelty=True) model saved to: {model_path}")
    print("You can now load this model for final test evaluation.")

if __name__ == "__main__":
    main()
