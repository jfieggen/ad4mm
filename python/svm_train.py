#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from joblib import dump

def ocsvm_auc_score(estimator, X, y):
    """
    Custom scoring function for One-Class SVM.
    The decision_function gives a 'distance' from the hyperplane:
      - Positive => inlier
      - Negative => outlier
    We flip the sign so that higher scores => more likely outlier,
    which aligns with y=1 meaning 'outlier'.
    Then compute ROC AUC comparing these scores to y.
    """
    decision_scores = estimator.decision_function(X)  # higher => inlier
    outlier_scores = -decision_scores                 # flip sign => higher => outlier
    return roc_auc_score(y, outlier_scores)

def main():
    # 1) Load your data
    data_dir = "/well/clifton/users/ncu080/ad4mm/data"
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    print("Loading training data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    
    # If y_train is a single-column DataFrame, convert it to a Series
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Unique labels in y_train:", y_train.unique())

    # 2) Define One-Class SVM with a placeholder
    oc_svm = OneClassSVM()

    # 3) Define the parameter grid
    # Fixed nu=0.003, vary kernel and gamma
    param_grid = {
        'nu': [0.003],           # fix nu to 0.003 as known in advance
        'kernel': ['rbf'],       # typically 'rbf' is standard for one-class
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }

    # 4) Custom scorer
    ocsvm_scorer = make_scorer(ocsvm_auc_score, greater_is_better=True)

    # 5) Stratified CV (makes sense if y is 0/1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 6) GridSearchCV
    grid_search = GridSearchCV(
        estimator=oc_svm,
        param_grid=param_grid,
        scoring=ocsvm_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting Grid Search for One-Class SVM...")
    grid_search.fit(X_train, y_train)
    print("Grid Search complete.")
    print("Best parameters found:", grid_search.best_params_)
    print("Best CV AUC:", grid_search.best_score_)

    # 7) Evaluate on entire training set
    best_oc_svm = grid_search.best_estimator_

    # One-Class SVM output: +1 => inlier, -1 => outlier
    y_pred_oc = best_oc_svm.predict(X_train)
    # Map to your labeling: 1 => outlier, 0 => inlier
    y_pred_mapped = [1 if p == -1 else 0 for p in y_pred_oc]

    # For ROC AUC
    decision_scores = best_oc_svm.decision_function(X_train)
    outlier_scores = -decision_scores  # flip sign so higher => outlier

    # 8) Classification metrics if y_train is binary
    if set(y_train.unique()) <= {0, 1}:
        print("\nClassification Report (treating 1 as anomaly):")
        print(classification_report(y_train, y_pred_mapped, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_train, y_pred_mapped))

        auc_value = roc_auc_score(y_train, outlier_scores)
        print(f"Training ROC AUC with best model: {auc_value:.3f}")
    else:
        print("\nNote: y_train not strictly 0/1; skipping classification metrics.")

    # 9) Save the best model
    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_one_class_svm.joblib")
    dump(best_oc_svm, model_path)
    print(f"Best One-Class SVM model saved to: {model_path}")

if __name__ == "__main__":
    main()
