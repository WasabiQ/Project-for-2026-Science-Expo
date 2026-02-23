import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- CONFIGURATION ---
MODEL_PATH = "skynet_brain.pth"
SCALER_FILE = "scaler_params.json"
VAULT_FILE = "chemical_vault.json"
DATA_FILE = "Tox21.csv"

def log_msg(msg):
    # Logs to stderr so CypherUI doesn't try to parse them
    sys.stderr.write(f"[SKYNET_REINFORCED] {msg}\n")
    sys.stderr.flush()

# --- REINFORCED NEURAL ARCHITECTURE ---
# Uses 150 neurons with high-gradient LeakyReLU for aggressive pattern matching
class ToxNet(nn.Module):
    def __init__(self, input_size=11):
        super().__init__()
        # Layer 1: 80 Neurons (Feature Extraction)
        self.fc1 = nn.Linear(input_size, 80)
        # Layer 2: 50 Neurons (Policy Evaluation)
        self.fc2 = nn.Linear(80, 50)
        # Layer 3: 20 Neurons (Advantage Estimation)
        self.fc3 = nn.Linear(50, 20)
        # Output: 1 Neuron (Toxicity Verdict)
        self.fc4 = nn.Linear(20, 1)
        
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # Using 0.01 negative slope for reinforced gradient flow
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        return torch.sigmoid(self.fc4(x))

# --- REINFORCED TRAINING MODULE ---
def train_system():
    log_msg("Initiating Reinforced Training...")
    if not os.path.exists(DATA_FILE):
        log_msg("FATAL: Tox21.csv missing.")
        return

    df = pd.read_csv(DATA_FILE)
    cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
            'SR-HSE', 'SR-MMP']
    
    X = df[cols].fillna(0).values
    y = df.iloc[:, -1].fillna(0).values.reshape(-1, 1)

    # Scaling to minimize Loss
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_scaled = (X - mean) / std

    with open(SCALER_FILE, 'w') as f:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)

    model = ToxNet()
    # AdamW Optimizer for better regularization during reinforcement
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    X_t, y_t = torch.FloatTensor(X_scaled), torch.FloatTensor(y)

    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            log_msg(f"Reinforcement Epoch {epoch} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    log_msg("Model Reinforced and Saved.")

# --- DUAL-SOURCE SEARCH & INFERENCE ---
def run_scan(name):
    query = name.lower().strip()
    features = None

    # 1. CHECK CSV FIRST (Primary)
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            # Find matching name in column 0
            match = df[df.iloc[:, 0].str.lower() == query]
            if not match.empty:
                features = match.iloc[0, 1:12].fillna(0).values.astype(float)
                log_msg(f"Source: Tox21 Textbook")
        except: pass

    # 2. CHECK VAULT SECOND (Secondary)
    if features is None and os.path.exists(VAULT_FILE):
        try:
            with open(VAULT_FILE, 'r') as f:
                vault = json.load(f)
                if query in vault:
                    features = np.array(vault[query]['markers'])
                    log_msg(f"Source: Chemical Vault")
        except: pass

    # 3. IF NOT FOUND IN BOTH: OUTPUT "NOT FOUND"
    if features is None:
        print("NOT FOUND")
        return

    # 4. PERFORM REINFORCED INFERENCE
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_FILE)):
        print("ERROR:SYSTEM_OFFLINE")
        return

    with open(SCALER_FILE, 'r') as f:
        p = json.load(f)
        m, s = np.array(p['mean']), np.array(p['std'])

    model = ToxNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    start = time.perf_counter()
    with torch.no_grad():
        scaled = (features - m) / s
        tensor_in = torch.FloatTensor(scaled).unsqueeze(0)
        prob = model(tensor_in).item()
    
    dur = (time.perf_counter() - start) * 1000
    verdict = "HAZARDOUS" if prob > 0.5 else "SAFE"
    
    # FINAL OUTPUT TO CYPHERUI.GO
    print(f"RESULT:{verdict}:{prob:.4f}")
    log_msg(f"Inference: {dur:.4f}ms")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train": train_system()
        else: run_scan(sys.argv[1])
    else:
        log_msg("Mode: Scan (Default)")
        run_scan(input("Chemical: "))