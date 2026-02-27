import os
import sys
import time
import requests
import skynet_pb2
import pubchempy as pcp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from chembl_webresource_client.new_client import new_client
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.protobuf.timestamp_pb2 import Timestamp

# =============================================================================
# SKYNET VAULT CORE - V12.0.5 (TITAN_ENGINE)
# =============================================================================

class SkynetTitan:
    def __init__(self, bin_path="chemical_vault.bin", input_path="compounds.txt"):
        self.bin_path = bin_path
        self.input_path = input_path
        self.vault = skynet_pb2.Vault()
        self.fp_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        self._load_vault()

    def _load_vault(self):
        """Native binary ingestion."""
        if os.path.exists(self.bin_path):
            with open(self.bin_path, "rb") as f:
                self.vault.ParseFromString(f.read())

    def _save_vault(self):
        """Atomic filesystem commit."""
        temp = f"{self.bin_path}.tmp"
        with open(temp, "wb") as f:
            f.write(self.vault.SerializeToString())
        os.replace(temp, self.bin_path)

    def scrape_node(self, name):
        """High-bandwidth identifier extraction."""
        res = {
            "name": name,
            "smiles": None,
            "iupac": None,
            "chembl_id": None,
            "cid": 0,
            "props": {}
        }
        
        # --- CHANNELS: NCI & ChEMBL ---
        try:
            # Parallel check 1: NCI Cactus
            c_url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/"
            r_s = requests.get(c_url + "smiles", timeout=7)
            if r_s.status_code == 200: res["smiles"] = r_s.text.strip()
            
            r_i = requests.get(c_url + "iupac_name", timeout=7)
            if r_i.status_code == 200: res["iupac"] = r_i.text.strip()
        except: pass

        try:
            # Parallel check 2: ChEMBL API
            mol_query = new_client.molecule.filter(molecule_synonyms__molecule_synonym__iexact=name)
            if mol_query:
                m = mol_query[0]
                res["chembl_id"] = m['molecule_chembl_id']
                if not res["smiles"]: res["smiles"] = m['molecule_structures']['canonical_smiles']
        except: pass

        # --- CHANNEL: PUBCHEM (Full Descriptor Sweep) ---
        if res["smiles"]:
            try:
                p_comp = pcp.get_compounds(res["smiles"], "smiles")[0]
                res["cid"] = int(p_comp.cid or 0)
                res["iupac"] = res["iupac"] or p_comp.iupac_name
                res["props"] = {
                    "mw": float(p_comp.molecular_weight or 0),
                    "logp": float(p_comp.xlogp or 0),
                    "tpsa": float(p_comp.tpsa or 0),
                    "hbd": int(p_comp.h_bond_donors or 0),
                    "hba": int(p_comp.h_bond_acceptors or 0),
                    "rb": int(p_comp.rotatable_bonds or 0),
                    "charge": float(p_comp.formal_charge or 0),
                    "complexity": float(p_comp.complexity or 0)
                }
            except: pass

        return res

    def induct_into_vault(self, data):
        """Binary packing logic for the Protobuf Map."""
        if not data["smiles"]: return
        
        chem = self.vault.entries[data["name"].lower()]
        chem.name = data["name"]
        chem.smiles = data["smiles"]
        if data["iupac"]: chem.iupac_name = data["iupac"]
        if data["chembl_id"]: chem.chembl_id = data["chembl_id"]
        if data["cid"]: chem.pubchem_cid = data["cid"]

        # Neural Projection Layer
        mol = Chem.MolFromSmiles(data["smiles"])
        if mol:
            bv = self.fp_gen.GetCountFingerprintAsBitVect(mol)
            # Efficient bit-stream extension
            chem.fingerprint.extend([int(b) for b in bv.ToBitString()])

        # Physico-Chemical Descriptor Layer
        p = data["props"]
        if p:
            d = chem.descriptors
            d.molecular_weight = p.get("mw", 0)
            d.logp = p.get("logp", 0)
            d.tpsa = p.get("tpsa", 0)
            d.h_bond_donors = p.get("hbd", 0)
            d.h_bond_acceptors = p.get("hba", 0)
            d.rotatable_bonds = p.get("rb", 0)
            d.formal_charge = p.get("charge", 0)
            d.complexity = p.get("complexity", 0)

        chem.scraped_at.GetCurrentTime()
        chem.data_version = "Skynet-Titan-v12"

    def run(self, threads=8):
        if not os.path.exists(self.input_path): return
        with open(self.input_path, "r") as f:
            names = list(set([l.strip() for l in f if l.strip()]))
        
        todo = [n for n in names if n.lower() not in self.vault.entries]
        
        # Producer-Consumer Multithreading
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_name = {executor.submit(self.scrape_node, n): n for n in todo}
            
            processed = 0
            for future in tqdm(as_completed(future_to_name), total=len(todo), desc="TITAN_VAULT"):
                try:
                    data = future.result()
                    self.induct_into_vault(data)
                    processed += 1
                    
                    # Atomic commit every 25 successful inductions
                    if processed % 25 == 0:
                        self._save_vault()
                except: continue

        self._save_vault()

if __name__ == "__main__":
    SkynetTitan().run()