# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# =============================================================================
# LOGGING
# =============================================================================
def log(tag, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    sys.stderr.write(f"[{timestamp}] [{tag}] {message}\n")
    sys.stderr.flush()

# =============================================================================
# ATTENTION MODULE
# =============================================================================
class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

# =============================================================================
# NEURON / RESIDUAL UNIT
# =============================================================================
class NeuralNerve(nn.Module):
    """
    200-Neuron Residual Block with:
    - GroupNorm (stable on small batches)
    - GELU
    - Squeeze-Excitation
    """
    def __init__(self, size=200):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, size)
        self.fc1 = nn.Linear(size, size)
        self.gn2 = nn.GroupNorm(8, size)
        self.fc2 = nn.Linear(size, size)
        self.se  = SqueezeExcitation(size)

    def forward(self, x):
        identity = x
        out = self.fc1(F.gelu(self.gn1(x)))
        out = self.fc2(F.gelu(self.gn2(out)))
        out = self.se(out)
        return out + identity

# =============================================================================
# MODEL
# =============================================================================
class SkynetArchitecture(nn.Module):
    def __init__(self, input_dim=11, core_size=200):
        super().__init__()

        self.projection = nn.Linear(input_dim, 2048)
        self.compressor = nn.Linear(2048, core_size)

        self.core = nn.Sequential(
            *[NeuralNerve(core_size) for _ in range(10)]
        )

        self.head = nn.Sequential(
            nn.Linear(core_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.gelu(self.projection(x))
        x = F.gelu(self.compressor(x))
        x = self.core(x)
        return self.head(x)

# =============================================================================
# GRADIENT CENTRALIZATION
# =============================================================================
def centralize_gradients(model):
    for p in model.parameters():
        if p.grad is not None and p.ndim > 1:
            p.grad.data -= p.grad.data.mean(
                dim=tuple(range(1, p.ndim)),
                keepdim=True
            )

# =============================================================================
# FEATURES
# =============================================================================
FEATURES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP"
]

# =============================================================================
# TRAINING ENTRY
# =============================================================================
def run_induction(epochs=10, batch_size=128):
    log("SYSTEM", "Initializing Singularity Engine")

    if not os.path.exists("Tox21.csv"):
        log("CRITICAL", "Tox21.csv not found")
        return

    df = pd.read_csv("Tox21.csv")

    X = df[FEATURES].fillna(0).values.astype(np.float32)
    y = df.iloc[:, -1].fillna(0).values.astype(np.float32).reshape(-1, 1)

    # Normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    X = torch.tensor(X)
    y = torch.tensor(y)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkynetArchitecture(input_dim=X.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(loader)
    )

    criterion = nn.BCELoss()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                preds = model(xb)
                loss = criterion(preds, yb)

            scaler.scale(loss).backward()
            centralize_gradients(model)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        log("EPOCH", f"{epoch+1}/{epochs} | loss={total_loss/len(loader):.6f}")

    log("SYSTEM", "Training complete")
    return model