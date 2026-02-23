import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

# --- CONSTANTS ---
PROJECT_NAME = "ToxNet"
MODEL_PATH = "toxnet_brain.pth"
CACHE_FILE = "tox_cache.json"
SCALER_FILE = "scaler_params.json"
DATA_FILE = "Tox21.csv" # Your 7,832 chemical subset

# --- THE 150-NEURON DEEP ARCHITECTURE ---
class ToxNet(nn.Module):
    def __init__(self, input_size=11):
        super(ToxNet, self).__init__()
        # Architecture: 80 -> 50 -> 20 (150 Total Neurons)
        self.fc1 = nn.Linear(input_size, 80) 
        self.fc2 = nn.Linear(80, 50)         
        self.fc3 = nn.Linear(50, 20)         
        self.fc4 = nn.Linear(20, 1)          
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

# --- IONISATION ENTHALPY TRAINING ---
def train_model():
    print(f"[{PROJECT_NAME}] Analysing 7,832 Compounds...")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found!")
        return

    df = pd.read_csv(DATA_FILE)
    
    # 11 Markers used to predict the 12th (SR-p53)
    cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
            'SR-HSE', 'SR-MMP']
    
    X = df[cols].fillna(0).values
    y = df['SR-p53'].fillna(0).values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save Scaler for the Collider Inference
    with open(SCALER_FILE, 'w') as f:
        json.dump({'mean': scaler.mean_.tolist(), 'std': scaler.scale_.tolist()}, f)

    model = ToxNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    X_t, y_t = torch.FloatTensor(X_scaled), torch.FloatTensor(y)
    
    print(f"[{PROJECT_NAME}] Commencing High-Energy Training...")
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[{PROJECT_NAME}] Ionisation Scaler & Brain Saved Successfully.")

# --- THE COLLIDER SCAN (Inference) ---
def run_scan(name, smiles, raw_data):
    # 1. Quick Cache Check
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            if name.lower() in cache:
                c = cache[name.lower()]
                print(f"CACHE_HIT:{c['result']}:{c['score']}")
                return

    # 2. Enthalpy Scaling
    if not os.path.exists(SCALER_FILE):
        print("ERROR: Scaler JSON missing. Run (T)rain first.")
        return
        
    with open(SCALER_FILE, 'r') as f:
        params = json.load(f)
        means = np.array(params['mean'])
        stds = np.array(params['std'])

    # 3. Neural Prediction
    model = ToxNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        # Manual Standardisation to match Scikit-Learn logic
        inputs = np.array(raw_data)
        scaled_inputs = (inputs - means) / (stds + 1e-8)
        
        tensor_in = torch.FloatTensor(scaled_inputs).unsqueeze(0)
        score = model(tensor_in).item()

    result = "TOXIC" if score > 0.5 else "SAFE"
    
    # 4. Save to Cache
    new_entry = {name.lower(): {"smiles": smiles, "result": result, "score": round(score, 4)}}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f: data = json.load(f)
        data.update(new_entry)
    else: data = new_entry
    
    with open(CACHE_FILE, 'w') as f: json.dump(data, f, indent=4)

    # 5. Output for Go UI
    print(f"RESULT:{result}:{score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 3:
        # Args: [Name] [SMILES] [11 Markers...]
        try:
            name_in = sys.argv[1]
            smi_in = sys.argv[2]
            feats = [float(x) for x in sys.argv[3:]]
            run_scan(name_in, smi_in, feats)
        except Exception as e:
            print(f"ERROR:{e}")
    else:
        print(f"[{PROJECT_NAME}] Terminal Control Online.")
        mode = input("Train or Scan? ").lower()
        if mode == 'train':
            train_model()
        else:
            # Dummy scan for a balanced molecular state
            run_scan("TestCompound", "N/A", [0.1, 0.0, 0.2, 0.0, 0.1, 0.5, 0.0, 0.1, 0.2, 0.0, 0.1])