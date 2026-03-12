import os # File path management for vault.bin
import sys # System-level operations
import math # Mathematical constants for physics layers
import time # Telemetry and performance profiling
import torch # Core AI framework
import torch.nn as nn # Neural network layers
import torch.nn.functional as F # Logic and activation functions
import pandas as pd # Dataframe management
import numpy as np # Matrix math for chemical fingerprints

from datetime import datetime # Logging timestamps
from torch.optim.lr_scheduler import OneCycleLR # LR heartbeat for least loss
from torch.cuda.amp import autocast, GradScaler # Mixed-precision stability
from torch_geometric.nn import EGNNConv, TransformerConv, global_mean_pool # Graph AI Core
from torch_geometric.data import Data, DataLoader # Graph data structures
from rdkit import Chem # Chemistry engine for SMILES
from rdkit.Chem import AllChem # 3D coordinate generation

# =============================================================================
# 1. THE TOX21 RECEPTOR MAP (The 12 Markers)
# =============================================================================
RECEPTORS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# =============================================================================
# 2. BINARY CONVERSION: SMILES -> GRAPH TENSORS
# =============================================================================
def smiles_to_binary_graph(smiles, target_array=None):
    mol = Chem.MolFromSmiles(smiles) # Parse raw SMILES string
    if not mol: return None
    mol = Chem.AddHs(mol) # Add hydrogens for realistic 3D volume
    AllChem.EmbedMolecule(mol, AllChem.ETKDG()) # Generate 3D spatial positions
    
    # Node Features: [Atomic#, Degree, Charge, Hybridization, Aromaticity]
    nodes = [[a.GetAtomicNumber(), a.GetDegree(), a.GetFormalCharge(), 
              float(a.GetHybridization()), float(a.GetIsAromatic())] for a in mol.GetAtoms()]
    x = torch.tensor(nodes, dtype=torch.float)
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float) # 3D Math
    
    # Edge Features: Bond connectivity and Bond Type
    edges, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        edge_attr += [[float(b.GetBondTypeAsDouble())]] * 2
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Line Graph (Bond-to-Bond connectivity) to prevent data deterioration
    line_edges = []
    bonds = list(mol.GetBonds())
    for i in range(len(bonds)):
        for j in range(i + 1, len(bonds)):
            if set([bonds[i].GetBeginAtomIdx(), bonds[i].GetEndAtomIdx()]) & \
               set([bonds[j].GetBeginAtomIdx(), bonds[j].GetEndAtomIdx()]):
                line_edges += [[i, j], [j, i]]
    line_idx = torch.tensor(line_edges, dtype=torch.long).t().contiguous()

    y = torch.tensor([target_array], dtype=torch.float) if target_array is not None else None
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, line_edge_index=line_idx, y=y)

# =============================================================================
# 3. CORE HYBRID ENGINE (Upgraded Architecture)
# =============================================================================
class HybridNerve(nn.Module):
    def __init__(self, size=200, edge_dim=1):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, size) # Stable norm for small chemical batches
        self.egnn = EGNNConv(size, size, edge_dim, m_dim=size) # Spatial eye
        self.gn2 = nn.GroupNorm(8, size)
        self.atom_transformer = TransformerConv(size, size // 4, heads=4, edge_dim=edge_dim)
        self.bond_transformer = TransformerConv(edge_dim, edge_dim // 2, heads=2) # Line Graph

    def forward(self, x, pos, edge_index, edge_attr, line_edge_index):
        identity = x # Residual link prevents vanishing gradients
        x_s, pos_up = self.egnn(F.gelu(self.gn1(x)), pos, edge_index, edge_attr)
        edge_up = self.bond_transformer(edge_attr, line_edge_index)
        x_f = self.atom_transformer(F.gelu(self.gn2(x_s)), edge_index, edge_up)
        return x_f + identity, pos_up, edge_up

class SkynetArchitecture(nn.Module):
    def __init__(self, input_dim=5, core_size=200):
        super().__init__()
        self.projection = nn.Linear(input_dim, 2048) # Wide entry
        self.compressor = nn.Linear(2048, core_size) # Compression
        self.core = nn.ModuleList([HybridNerve(core_size) for _ in range(10)]) # 10 Layer Depth
        self.head = nn.Sequential(
            nn.Linear(core_size, 128),
            nn.GELU(),
            nn.Linear(128, 12), # Predicts 12 Tox21 receptors simultaneously
            nn.Sigmoid() # Probability math: 1 / (1 + exp(-x))
        )

    def forward(self, data):
        x, pos, edge_idx, edge_at, line_idx = data.x, data.pos, data.edge_index, data.edge_attr, data.line_edge_index
        x = F.gelu(self.projection(x))
        x = F.gelu(self.compressor(x))
        for layer in self.core:
            x, pos, edge_at = layer(x, pos, edge_idx, edge_at, line_idx)
        x = global_mean_pool(x, data.batch) # Condense molecule to vector
        return self.head(x)

# =============================================================================
# 4. TRAINING: THE INDUCTION PULSE
# =============================================================================
def run_induction(epochs=50, batch_size=32):
    log("SYSTEM", "Starting Least-Loss Induction on 12 Markers")
    
    # Load data from CSV (The output of your Rust scraper's vault conversion)
    df = pd.read_csv("Tox21.csv")
    dataset = [smiles_to_binary_graph(s, row[RECEPTORS].values) for s, row in df.iterrows()]
    loader = DataLoader([d for d in dataset if d], batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkynetArchitecture().to(device)
    
    # AdamW + OneCycleLR for the global minimum loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(loader))
    criterion = nn.BCELoss() # Binary Cross-Entropy math for the 12 receptors
    scaler = GradScaler(enabled=torch.cuda.is_available())

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                preds = model(batch)
                loss = criterion(preds, batch.y) # Multi-task loss math
            
            scaler.scale(loss).backward()
            centralize_gradients(model) # Keep weights stable
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        
        log("EPOCH", f"{epoch+1}/{epochs} | Loss: {total_loss/len(loader):.8f}")
    
    return model
