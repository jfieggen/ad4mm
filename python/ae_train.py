#!/usr/bin/env python3
import os
os.environ["TORCH_DYNAMO_DISABLE"] = "1"  # Disable torch-dynamo as doesn't work with python 3.11
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

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

    # 4) Convert data to PyTorch tensors and create a DataLoader
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_train_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 5) Set up device: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 6) Define a simple fully connected autoencoder
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=32):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon

    model = Autoencoder(input_dim=input_dim, latent_dim=32).to(device)
    print(model)

    # 7) Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    # 8) Training loop
    print("Starting training for {} epochs...".format(num_epochs))
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(dataset)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print("Epoch [{}/{}], Loss: {:.6f}".format(epoch+1, num_epochs, epoch_loss))
    print("Training complete.")

    # 9) Evaluate the model on the training data
    model.eval()
    with torch.no_grad():
        X_train_tensor = X_train_tensor.to(device)
        recon = model(X_train_tensor)
        # Compute mean squared error per sample (reconstruction error)
        mse = torch.mean((recon - X_train_tensor) ** 2, dim=1)
        reconstruction_errors = mse.cpu().numpy()

    # 10) If y_train is binary, compute ROC AUC and classification metrics
    if set(np.unique(y_train)) <= {0, 1}:
        auc = roc_auc_score(y_train, reconstruction_errors)
        print("Training ROC AUC (using reconstruction error): {:.3f}".format(auc))
        # Set a threshold based on a chosen percentile (e.g., top 5% reconstruction errors as anomalies)
        threshold = np.percentile(reconstruction_errors, 95)
        y_pred = (reconstruction_errors > threshold).astype(int)
        print("\nClassification Report:")
        print(classification_report(y_train, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_train, y_pred))
    else:
        print("y_train is not binary; skipping classification metrics.")

    # 11) Save the trained model and scaler for later use
    model_dir = "/well/clifton/users/ncu080/ad4mm/outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "autoencoder_model.pth")
    scaler_path = os.path.join(model_dir, "autoencoder_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print("Model saved to:", model_path)
    print("Scaler saved to:", scaler_path)

if __name__ == "__main__":
    main()
