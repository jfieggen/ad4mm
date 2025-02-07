#!/usr/bin/env python3
import os
# Disable Torch Dynamo and Compilation
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Ensure distributed debug level is set correctly
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"  # Must be "OFF", "INFO", or "DETAIL"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

def main():
    # 1) Define paths to your training data
    data_dir = "/well/clifton/users/ncu080/ad4mm/data"
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    # 2) Load the training data
    print("Loading training data...")
    X_train_df = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)
    
    # Convert y_train to a 1D array (if a single-column DataFrame)
    if y_train_df.shape[1] == 1:
        y_train = y_train_df.iloc[:, 0].values
    else:
        y_train = y_train_df.values.ravel()

    # 3) Preprocess X_train: Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    input_dim = X_train_scaled.shape[1]

    # Save full training data for evaluation later
    X_train_all = X_train_scaled.copy()

    # --- Filter for majority class only (class 0) for training ---
    mask_majority = (y_train == 0)
    if np.sum(mask_majority) == 0:
        raise ValueError("No samples found for the majority class (label 0).")
    X_train_majority = X_train_scaled[mask_majority]
    print("Majority class samples (0):", X_train_majority.shape[0])

    # 4) Split majority-class data into training and validation sets (80%/20% split)
    X_train_majority_split, X_val_majority_split = train_test_split(
        X_train_majority, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors and create DataLoaders
    X_train_tensor = torch.tensor(X_train_majority_split, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_majority_split, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 5) Set up device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 6) Define a simple fully connected autoencoder
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
            x_recon = self.decoder(z)
            return x_recon

    model = Autoencoder(input_dim=input_dim, latent_dim=4).to(device)
    print(model)

    # 7) Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    # Lists to store loss values for plotting training and validation curves
    train_losses = []
    val_losses = []

    # 8) Training loop with validation (using only majority-class samples)
    print("Starting training for {} epochs...".format(num_epochs))
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0].to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                running_val_loss += loss.item() * batch_x.size(0)
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("Epoch [{}/{}], Train Loss: {:.6f}, Val Loss: {:.6f}".format(
                epoch + 1, num_epochs, epoch_train_loss, epoch_val_loss))
    print("Training complete.")

    # 8.5) Plot training and validation loss curves
    training_plot_dir = "/well/clifton/users/ncu080/ad4mm/outputs/training"
    os.makedirs(training_plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    loss_curve_path = os.path.join(training_plot_dir, "training_validation_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    print("Training and validation curves saved to:", loss_curve_path)

    # 9) Evaluate the model on the full training data (all classes) for anomaly detection metrics
    model.eval()
    X_train_all_tensor = torch.tensor(X_train_all, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_train_all_tensor)
        # Compute mean squared error per sample (reconstruction error)
        mse = torch.mean((recon - X_train_all_tensor) ** 2, dim=1)
        reconstruction_errors = mse.cpu().numpy()

    # 9.1) Print summary statistics on the reconstruction error for both classes
    print("\nReconstruction Error Summary Statistics:")
    for class_label in [0, 1]:
        mask_class = (y_train == class_label)
        errors_class = reconstruction_errors[mask_class]
        if len(errors_class) == 0:
            print(f"No samples found for class {class_label}.")
            continue
        stats = {
            "mean": np.mean(errors_class),
            "std": np.std(errors_class),
            "min": np.min(errors_class),
            "25th percentile": np.percentile(errors_class, 25),
            "median": np.median(errors_class),
            "75th percentile": np.percentile(errors_class, 75),
            "max": np.max(errors_class)
        }
        print(f"Class {class_label}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")
    
    # 9.2) Plot the distributions of the reconstruction losses for both classes
    plt.figure(figsize=(10, 6))
    bins = 50
    plt.hist(reconstruction_errors[y_train == 0], bins=bins, alpha=0.5, label="Class 0")
    plt.hist(reconstruction_errors[y_train == 1], bins=bins, alpha=0.5, label="Class 1")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Reconstruction Errors")
    plt.legend()
    dist_plot_path = os.path.join(training_plot_dir, "reconstruction_error_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    print("Reconstruction error distribution plot saved to:", dist_plot_path)

    # 10) Determine the optimal threshold using ROC analysis (maximizing tpr - fpr)
    if set(np.unique(y_train)) <= {0, 1}:
        # Compute ROC curve
        fpr, tpr, thresholds_roc = roc_curve(y_train, reconstruction_errors)
        optimal_threshold_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_roc[optimal_threshold_idx]
        print("\nOptimal threshold based on maximizing (TPR - FPR): {:.6f}".format(optimal_threshold))
        
        # Optionally, you can compute predictions and display further metrics:
        y_pred = (reconstruction_errors > optimal_threshold).astype(int)
        auc = roc_auc_score(y_train, reconstruction_errors)
        print("\nTraining ROC AUC (using reconstruction error): {:.3f}".format(auc))
        print("\nClassification Report:")
        print(classification_report(y_train, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_train, y_pred))
    else:
        print("y_train is not binary; skipping ROC thresholding and classification metrics.")

    # 11) Save the trained model and scaler for later use
    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "autoencoder_model.pth")
    scaler_path = os.path.join(model_dir, "autoencoder_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print("\nModel saved to:", model_path)
    print("Scaler saved to:", scaler_path)

if __name__ == "__main__":
    main()
