import sys
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import json

# Import your architecture from your Skynet.py
from Skynet import SkynetArchitecture, FEATURES, MAP

def get_binary_fingerprint(smiles):
    """Converts SMILES 'Letters' to 2048-bit Binary."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return torch.FloatTensor(list(fp))

def run_pipeline(chemical_name):
    # 1. Check local vault for SMILES
    # (Assuming your vault is a CSV for this script's simplicity)
    vault = pd.read_csv("chemical_vault.csv") 
    match = vault[vault['name'].str.contains(chemical_name, case=False)]
    
    if match.empty:
        return {"error": "Chemical not found in vault."}
    
    smiles = match.iloc[0]['smiles']

    # 2. Check Tox21 for real data
    tox_data = pd.read_csv("Tox21.csv")
    real_match = tox_data[tox_data['smiles'] == smiles]

    if not real_match.empty:
        source = "Experimental (Tox21)"
        results = real_match[FEATURES].iloc[0].to_dict()
    else:
        # 3. AI Prediction Mode
        source = "AI Prediction (Skynet v12)"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load your trained model
        model = SkynetArchitecture(input_dim=2048).to(device)
        # model.load_state_dict(torch.load("skynet_weights.pth")) # Load if you have it
        model.eval()

        binary_input = get_binary_fingerprint(smiles).to(device).unsqueeze(0)
        with torch.no_grad():
            preds = model(binary_input).squeeze().tolist()
        
        results = {FEATURES[i]: preds[i] for i in range(len(FEATURES))}

    # 4. Format for UI
    output = {
        "name": chemical_name,
        "smiles": smiles,
        "source": source,
        "markers": []
    }
    
    for marker, score in results.items():
        if score > 0.5: # Threshold for "Active"
            output["markers"].append(MAP[marker])

    return output

if __name__ == "__main__":
    query = sys.argv[1]
    print(json.dumps(run_pipeline(query)))