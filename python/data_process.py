import os
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_config(config_path="config.yml"):
    """
    Load the YAML configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def preprocess_data(df, categorical_columns=["Back Pain", "Chest Pain"],
                    drop_columns=["time", "PRELP", "PAMR1", "FCRL2", "IDS", "FRZB",
                                  "PTS", "CLEC4A", "FCRLB", "AMFR", "CD79B"],
                    target_column="Myeloma"):
    """
    Preprocess the dataframe:
      - Drop unwanted columns,
      - Separate the target variable,
      - Identify numeric features to scale (continuous features only),
      - Convert categorical columns to type 'category',
      - One-hot encode the categorical columns.
      
    Returns:
      X: Features dataframe after one-hot encoding.
      y: Target series.
      numeric_features: List of columns that were identified as continuous numeric features
                        and should be scaled.
    """
    print("\n=== Preprocessing Data ===")
    print("Original DataFrame shape:", df.shape)
    print("Original columns:", df.columns.tolist())
    
    # Ensure the drop and target columns exist
    for col in drop_columns + [target_column]:
        assert col in df.columns, f"Expected column '{col}' not found in the data."
    
    # Drop unwanted columns
    df = df.drop(columns=drop_columns, errors="ignore")
    print("After dropping columns {}:".format(drop_columns), df.shape)
    print("Columns now:", df.columns.tolist())
    
    # Separate target variable
    y = df[target_column]
    X = df.drop(columns=[target_column])
    print("Separated target column '{}'.".format(target_column))
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    
    # Identify continuous numeric features to scale.
    # We do not scale categorical_columns or the binary column "Sex".
    numeric_features = [col for col in X.columns if col not in (categorical_columns + ["Sex"])]
    print("Identified continuous numeric features to scale:", numeric_features)
    
    # Convert specified categorical columns to 'category' dtype
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category")
            print(f"Converted column '{col}' to 'category' dtype.")
    
    # One-hot encode the categorical columns.
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
    print("After one-hot encoding categorical columns:")
    print("Features shape:", X.shape)
    print("Columns now:", X.columns.tolist())
    
    return X, y, numeric_features

def run_tests(X_train, y_train, X_test, y_test, numeric_features):
    """
    Run tests to ensure:
      - The dropped columns and target are not present in the features.
      - Number of rows in features and target match.
      - Train and test feature sets have identical columns.
      - Print summary statistics of the scaled numeric features.
    """
    print("\n=== Running Tests ===")
    
    # Test that dropped columns are not present
    for df, name in zip([X_train, X_test], ["X_train", "X_test"]):
        assert "time" not in df.columns, f"'time' column was not dropped in {name}."
        for col in ["PRELP", "PAMR1", "FCRL2", "IDS", "FRZB", "PTS", 
                    "CLEC4A", "FCRLB", "AMFR", "CD79B"]:
            assert col not in df.columns, f"Column '{col}' was not dropped in {name}."
    
    # Test that target column is not present in features
    for df, name in zip([X_train, X_test], ["X_train", "X_test"]):
        assert "Myeloma" not in df.columns, f"Target column 'Myeloma' is still present in {name}."
    
    # Test row alignment between features and target
    assert len(X_train) == len(y_train), "Number of rows in X_train and y_train do not match."
    assert len(X_test) == len(y_test), "Number of rows in X_test and y_test do not match."
    
    # Test that training and test feature sets have identical columns
    assert set(X_train.columns) == set(X_test.columns), "Train and test features do not have the same columns."
    
    # Print summary statistics for scaled numeric features (from training data)
    print("\nSummary statistics for scaled numeric features in X_train:")
    if numeric_features:
        print(X_train[numeric_features].describe())
    else:
        print("No numeric features to scale.")
    
    print("All tests passed!")

def main():
    # Load configuration (update the path if needed)
    config = load_config("/well/clifton/users/ncu080/ad4mm/config.yml")
    
    # Get file paths from config
    train_file = config["data"]["train"]
    test_file = config["data"]["test"]
    output_dir = config["outputs"]["data"]
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading training data from: {train_file}")
    train_df = pd.read_csv(train_file)
    
    print(f"\nLoading test data from: {test_file}")
    test_df = pd.read_csv(test_file)
    
    # Preprocess training and test datasets
    categorical_columns = ["Back Pain", "Chest Pain"]
    drop_columns = ["time", "PRELP", "PAMR1", "FCRL2", "IDS", "FRZB",
                    "PTS", "CLEC4A", "FCRLB", "AMFR", "CD79B"]
    target_column = "Myeloma"
    
    print("\nPreprocessing training data...")
    X_train, y_train, numeric_features = preprocess_data(train_df,
                                                         categorical_columns,
                                                         drop_columns,
                                                         target_column)
    
    print("\nPreprocessing test data...")
    X_test, y_test, _ = preprocess_data(test_df,
                                        categorical_columns,
                                        drop_columns,
                                        target_column)
    
    # Align train and test feature columns (ensuring same dummy variables exist in both)
    print("\nAligning training and test feature columns...")
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
    print("After alignment:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    # Scale the continuous numeric features (fit on training, transform both)
    if numeric_features:
        print("\nScaling continuous numeric features:", numeric_features)
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
        print("After scaling, summary statistics for X_train numeric features:")
        print(pd.DataFrame(X_train[numeric_features]).describe())
    
    # Run tests
    run_tests(X_train, y_train, X_test, y_test, numeric_features)
    
    # Define output file paths
    x_train_file = os.path.join(output_dir, "x_train.csv")
    y_train_file = os.path.join(output_dir, "y_train.csv")
    x_test_file = os.path.join(output_dir, "x_test.csv")
    y_test_file = os.path.join(output_dir, "y_test.csv")
    
    # Save preprocessed datasets
    print("\nSaving preprocessed training and test data...")
    X_train.to_csv(x_train_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    X_test.to_csv(x_test_file, index=False)
    y_test.to_csv(y_test_file, index=False)
    
    print("\nData saved successfully:")
    print(f"  x_train: {x_train_file}")
    print(f"  y_train: {y_train_file}")
    print(f"  x_test:  {x_test_file}")
    print(f"  y_test:  {y_test_file}")

if __name__ == "__main__":
    main()
