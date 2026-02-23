import pubchempy as pcp
from chembl_webresource_client.new_client import new_client
import json
import time
import os

VAULT_FILE = "chemical_vault.json"
SAFE_DELAY = 0.4  # Keeps us under 3 requests/sec (PubChem limit is 5)

def load_vault():
    if os.path.exists(VAULT_FILE):
        with open(VAULT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_vault(vault):
    with open(VAULT_FILE, "w") as f:
        json.dump(vault, f, indent=4)

def fetch_all_sources(query):
    print(f"🔍 Researching: {query}...")
    try:
        # 1. PubChem: Get SMILES & Physical Descriptors
        compounds = pcp.get_compounds(query, "name")
        if not compounds:
            return None
        c = compounds[0]
        
        smiles = c.isomeric_smiles
        # Core descriptors for 150-neuron NN
        descriptors = [
            float(c.molecular_weight or 0),
            float(c.xlogp or 0),
            float(c.tpsa or 0),
            float(c.charge or 0),
            float(c.complexity or 0)
        ]

        # 2. ChEMBL: Get Bio-Target Predictions
        # This acts as your "Second Opinion" for the markers
        try:
            target_predictions = new_client.target_prediction
            preds = target_predictions.filter(smiles=smiles)
            # Filter for high-probability human protein targets
            bio_targets = [p['target_chembl_id'] for p in preds if float(p['probability']) > 0.7][:5]
        except:
            bio_targets = ["Offline/Error"]

        return {
            "smiles": smiles,
            "descriptors": descriptors,
            "chembl_targets": bio_targets,
            "source": "Aggregated"
        }

    except Exception as e:
        print(f"❌ Error fetching {query}: {e}")
        return None

def main(chemical_list):
    vault = load_vault()
    
    for chem in chemical_list:
        chem = chem.lower().strip()
        
        # Skip if already in vault (saves your API quota!)
        if chem in vault:
            continue
            
        result = fetch_all_sources(chem)
        
        if result:
            vault[chem] = result
            save_vault(vault) # Save every step so you don't lose data if it crashes
            print(f"✅ Saved to Vault: {chem}")
        
        # The "Anti-Ban" Sleep
        time.sleep(SAFE_DELAY)

if __name__ == "__main__":
    # Add your list of chemicals here or load from a TXT file
    demo_chemicals = ["Triclosan", "Glyphosate", "Bisphenol A", "Aspartame", "Caffeine"]
    main(demo_chemicals)