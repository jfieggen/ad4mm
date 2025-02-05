#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import dump

def main():
    # 1) Paths to train data
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

    # 3) Instantiate LOF with novelty=True (so it can be used later on new data)
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.003,  # example: ~0.3% outliers
        novelty=True,
        n_jobs=-1
    )

    print("\nFitting LOF on the TRAIN data...")
    # Fit the model on X_train
    lof.fit(X_train)

    # 4) Evaluate on TRAIN data
    # 'predict' => +1 for inliers, -1 for outliers
    y_pred_lof = lof.predict(X_train)

    # Map LOF predictions to your labeling scheme:
    # LOF: -1 => outlier, +1 => inlier
    # You:  1 => anomaly (outlier), 0 => normal (inlier)
    y_pred_mapped = [1 if val == -1 else 0 for val in y_pred_lof]

    # 5) Compute outlier scores on TRAIN
    # score_samples: higher => more inlier-like, so flip sign to get outlier-likelihood
    lof_raw_scores = lof.score_samples(X_train)
    outlier_scores = -lof_raw_scores  # higher => more outlier-like

    # 6) Classification metrics if y_train is 0=normal, 1=anomaly
    if set(y_train.unique()) <= {0, 1}:
        print("\n=== Classification Report (TRAIN) ===")
        print(classification_report(y_train, y_pred_mapped, zero_division=0))

        print("=== Confusion Matrix (TRAIN) ===")
        print(confusion_matrix(y_train, y_pred_mapped))

        train_auc = roc_auc_score(y_train, outlier_scores)
        print(f"TRAIN ROC AUC: {train_auc:.3f}")
    else:
        print("\nWarning: y_train not strictly {0, 1}. Skipping classification metrics.")

    # 7) Save the model for later use on test data
    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lof_novelty_model.joblib")

    dump(lof, model_path)
    print(f"\nLocal Outlier Factor (novelty=True) model saved to: {model_path}")

if __name__ == "__main__":
    main()
