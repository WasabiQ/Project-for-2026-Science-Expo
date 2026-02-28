import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

DATA_FILE = "Tox21.csv"
MODEL_FILE = "toxnet_model.pth"
SCALER_FILE = "scaler.json"

FEATURES = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP'
]
TARGET = 'SR-p53'

class ToxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main():
    """Main training function with proper error handling."""
    
    # FIX: Check if data file exists before attempting to load
    if not os.path.exists(DATA_FILE):
        log.error(f"Data file '{DATA_FILE}' not found!")
        log.error("Please ensure Tox21.csv is in the current directory.")
        sys.exit(1)
    
    try:
        log.info(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        log.info(f"Loaded {len(df)} rows")
        
    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        log.error(f"Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error loading data: {e}")
        sys.exit(1)

    # Validate required columns exist
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        log.error(f"Missing features in dataset: {missing_features}")
        sys.exit(1)
    
    if TARGET not in df.columns:
        log.error(f"Target column '{TARGET}' not found in dataset")
        sys.exit(1)

    try:
        # Data preparation
        X = df[FEATURES].fillna(0).values
        y = df[TARGET].fillna(0).values.reshape(-1, 1)
        
        log.info(f"Data shape: X={X.shape}, y={y.shape}")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler for later use
        try:
            with open(SCALER_FILE, "w") as f:
                json.dump({
                    "mean": scaler.mean_.tolist(),
                    "std": scaler.scale_.tolist()
                }, f, indent=4)
            log.info(f"Saved scaler parameters to {SCALER_FILE}")
        except Exception as e:
            log.warning(f"Could not save scaler: {e}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42
        )
        
        log.info(
            f"Train set: {X_train.shape[0]} samples, "
            f"Test set: {X_test.shape[0]} samples"
        )

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        
        model = ToxNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Training loop
        log.info("Starting training...")
        epochs = 100
        
        for epoch in range(epochs):
            # Train
            model.train()
            y_pred = model(X_train.to(device))
            loss = criterion(y_pred, y_train.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test.to(device))
                test_loss = criterion(y_test_pred, y_test.to(device))

            if epoch % 10 == 0:
                log.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Test Loss: {test_loss.item():.4f}"
                )

        # Save model
        try:
            torch.save(model.state_dict(), MODEL_FILE)
            log.info(f"Model saved to {MODEL_FILE}")
        except Exception as e:
            log.error(f"Error saving model: {e}")
            sys.exit(1)

        log.info("Training complete!")
        print("\n" + "="*50)
        print("MODEL + SCALER SAVED")
        print("="*50)

    except Exception as e:
        log.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
