import sys
import json
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from Skynet import SkynetArchitecture, FEATURES, MAP

def run_pipeline(chemical_name):
    try:
        # 1. SEARCH THE 7,832 ENTRIES
        tox_data = pd.read_csv("Tox21.csv")
        
        # Search by name
        match = tox_data[tox_data['compound_name'].str.contains(chemical_name, case=False, na=False)]
        
        # --- PATH A: DATABASE HIT ---
        if not match.empty:
            smiles = match.iloc[0]['smiles']
            # Gather the 11 real biological markers
            found = [MAP[f] for f in FEATURES if match.iloc[0][f] == 1]
            
            return {
                "name": chemical_name.upper(),
                "smiles": smiles,
                "source": "Tox21 Database",
                "markers": ["Yea, i found it and this is what it does:"] + found,
                "error": ""
            }

        # --- PATH B: NEURAL PREDICTION ---
        else:
            # If it's not in the CSV, we treat the input as a SMILES string to predict
            smiles_query = chemical_name 
            mol = Chem.MolFromSmiles(smiles_query)
            
            if not mol:
                return {"error": "Chemical not in Tox21 and input is not a valid SMILES for prediction."}

            # Prepare the 2048-bit Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            bits = torch.FloatTensor(list(fp))

            # Run the Skynet Architecture
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SkynetArchitecture(input_dim=2048).to(device)
            # model.load_state_dict(torch.load("skynet_weights.pth")) # Load your trained weights
            model.eval()

            with torch.no_grad():
                # The model outputs a probability (0.0 to 1.0)
                prediction = model(bits.to(device).unsqueeze(0)).item()

            # Determine "What it does" based on the AI's latent space projection
            status = "TOXIC_POTENTIAL_DETECTED" if prediction > 0.5 else "STABLE_STRUCTURE"
            
            return {
                "name": "AI_INFERENCE",
                "smiles": smiles_query,
                "source": "Skynet v12 (Neural)",
                "markers": [
                    "I used tox21 to predict what this does, take it back:",
                    f"Result: {status} (Neural Confidence: {prediction:.2f})"
                ],
                "error": ""
            }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Send the JSON signal back to Go
        print(json.dumps(run_pipeline(sys.argv[1])))