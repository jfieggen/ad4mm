#!/usr/bin/env python
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

def load_data(data_dir):
    """
    Load preprocessed training data.
    """
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    
    print(f"Loading training features from: {x_train_path}")
    print(f"Loading training target from: {y_train_path}")
    
    X_train = pd.read_csv(x_train_path)
    # Squeeze y_train into a Series (if y_train.csv is a single column)
    y_train = pd.read_csv(y_train_path).squeeze()
    return X_train, y_train

def main():
    # Define directories: change these paths as required.
    data_dir = "/well/clifton/users/ncu080/ad4mm/data"
    output_model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    
    # Ensure the output directory exists
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Load training data
    X_train, y_train = load_data(data_dir)
    
    # Set up a logistic regression estimator.
    # solver='liblinear' works with both 'l1' and 'l2' penalties.
    lr = LogisticRegression(solver='liblinear', max_iter=1000)
    
    # Define a grid of hyperparameters.
    # 'C' controls the inverse of regularization strength.
    # We'll search over both L1 and L2 penalties.
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }
    
    # Use StratifiedKFold to preserve the class distribution in each fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Set up GridSearchCV to search over the hyperparameters using ROC-AUC as the metric.
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting Grid Search for Logistic Regression...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest hyperparameters found:")
    print(grid_search.best_params_)
    print(f"Best ROC AUC score from CV: {grid_search.best_score_:.4f}")
    
    # Save the best estimator to a file for later use.
    model_file = os.path.join(output_model_dir, "logistic_regression_best_model.pkl")
    joblib.dump(grid_search.best_estimator_, model_file)
    print(f"Saved best model to: {model_file}")

if __name__ == "__main__":
    main()
