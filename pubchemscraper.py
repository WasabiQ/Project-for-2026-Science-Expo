# pubchemscraper.py
import pubchempy as pcp
import json
import os
import sys

VAULT_FILE = "chemical_vault.json"

DESCRIPTORS = [
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "Charge",
    "AtomStereoCount",
    "BondStereoCount",
    "IsotopeAtomCount"
]

def load_vault():
    if os.path.exists(VAULT_FILE):
        with open(VAULT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_vault(vault):
    with open(VAULT_FILE, "w") as f:
        json.dump(vault, f, indent=4)

def fetch_from_pubchem(query):
    compounds = pcp.get_compounds(query, "name")
    if not compounds:
        return None

    c = compounds[0]
    values = []

    for d in DESCRIPTORS:
        val = getattr(c, d, 0)
        if val is None:
            val = 0
        values.append(float(val))

    return values

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR:NO_INPUT")
        sys.exit(1)

    query = sys.argv[1].lower()
    vault = load_vault()

    # 1️⃣ Try local vault first
    if query in vault:
        print(" ".join(map(str, vault[query]["descriptors"])))
        sys.exit(0)

    # 2️⃣ Try PubChem (internet)
    try:
        desc = fetch_from_pubchem(query)
    except Exception:
        desc = None

    if desc is None:
        print("ERROR:OFFLINE_AND_NOT_CACHED")
        sys.exit(1)

    # 3️⃣ Save to vault
    vault[query] = {"descriptors": desc}
    save_vault(vault)

    print(" ".join(map(str, desc)))