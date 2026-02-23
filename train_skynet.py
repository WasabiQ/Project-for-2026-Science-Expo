# train_skynet.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Load dataset
df = pd.read_csv(DATA_FILE)
X = df[FEATURES].fillna(0).values
y = df[TARGET].fillna(0).values.reshape(-1, 1)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open(SCALER_FILE, "w") as f:
    json.dump({
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist()
    }, f, indent=4)

# Split
X_train, _, y_train, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Train
model = ToxNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

torch.save(model.state_dict(), MODEL_FILE)
print("MODEL + SCALER SAVED")