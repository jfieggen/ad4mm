#!/usr/bin/env python3
import os
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

def main():
    # Define the directory where preprocessed data is stored.
    data_dir = "/well/clifton/users/ncu080/ad4mm/data"
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    
    # Load training data
    print("Loading training data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    
    # In case y_train is a single-column DataFrame, convert it to a Series.
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    # Instantiate the XGBoost classifier with GPU optimization.
    model = XGBClassifier(
        tree_method='gpu_hist',        # Use GPU-accelerated tree building
        predictor='gpu_predictor',     # Use the GPU predictor
        use_label_encoder=False,       # Disable the deprecated label encoder
        eval_metric='auc'              # Evaluation metric for binary classification
    )
    
    # Define a hyperparameter grid.
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Set up GridSearchCV with 5-fold stratified cross-validation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',    # Use ROC AUC because of class imbalance
        cv=cv,
        verbose=1,
        n_jobs=-1             # Use all available cores
    )
    
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    print("\nGrid search complete.")
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation ROC AUC:", grid_search.best_score_)
    
    # Retrieve the best model and evaluate on the entire training set.
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    
    print("\nClassification Report on Training Data:")
    print(classification_report(y_train, y_train_pred))
    print("ROC AUC on Training Data:", roc_auc_score(y_train, y_train_proba))
    
    # Optionally, save the trained model to disk.
    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_xgboost_model.json")
    best_model.save_model(model_path)
    print(f"Best model saved to: {model_path}")

if __name__ == "__main__":
    main()
