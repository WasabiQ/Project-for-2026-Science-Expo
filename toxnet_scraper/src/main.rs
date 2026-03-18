use prost::Message;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File as StdFile;
use std::io::{Read};
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Semaphore;

pub mod skynet {
    include!(concat!(env!("OUT_DIR"), "/skynet_proto.rs"));
}

struct TitanScraper {
    client: Client,
    semaphore: Arc<Semaphore>,
}

impl TitanScraper {
    fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("ToxNet/Alpha")
                .build()
                .unwrap(),
            semaphore: Arc::new(Semaphore::new(8)),
        }
    }

    async fn fetch_pubchem(&self, name: &str) -> Option<(i32, String)> {
        let url = format!(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON",
            name
        );

        let resp = self.client.get(url).send().await.ok()?;
        let json: Value = resp.json().await.ok()?;

        let prop = &json["PropertyTable"]["Properties"][0];
        let cid = prop["CID"].as_i64()? as i32;
        let smiles = prop["CanonicalSMILES"].as_str()?.to_string();

        Some((cid, smiles))
    }

    async fn fetch_cactus_smiles(&self, name: &str) -> Option<String> {
        let url = format!(
            "https://cactus.nci.nih.gov/chemical/structure/{}/smiles",
            name
        );

        let resp = self.client.get(url).send().await.ok()?;
        let text = resp.text().await.ok()?;

        if text.trim().is_empty() {
            None
        } else {
            Some(text.trim().to_string())
        }
    }

    async fn fetch_chembl_id(&self, name: &str) -> Option<String> {
        let url = format!(
            "https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={}",
            name
        );

        let resp = self.client.get(url).send().await.ok()?;
        let text = resp.text().await.ok()?;

        // crude parsing (Alpha stage)
        if text.contains("CHEMBL") {
            let start = text.find("CHEMBL")?;
            Some(text[start..start + 12].to_string())
        } else {
            None
        }
    }

    async fn scrape_chemical(&self, name: String) -> Option<skynet::Chemical> {
        let _permit = self.semaphore.acquire().await.ok()?;

        let name_clean = name.to_lowercase();

        // 1. PubChem primary
        let (cid, smiles) = match self.fetch_pubchem(&name_clean).await {
            Some(data) => data,
            None => {
                // fallback to CACTUS
                let smiles = self.fetch_cactus_smiles(&name_clean).await?;
                (0, smiles)
            }
        };

        // 2. ChEMBL enrichment
        let chembl_id = self.fetch_chembl_id(&name_clean).await.unwrap_or_default();

        Some(skynet::Chemical {
            name: name_clean,
            pubchem_cid: cid,
            smiles,
            chembl_id,
            toxicity: String::new(), // placeholder for future
            ..Default::default()
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scraper = Arc::new(TitanScraper::new());
    let mut vault = skynet::Vault::default();
    let mut seen = HashSet::new();

    // Load existing vault
    if let Ok(mut file) = StdFile::open("chemical_vault.bin") {
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if let Ok(decoded) = skynet::Vault::decode(&buf[..]) {
            seen.extend(decoded.entries.keys().cloned());
            vault = decoded;
            println!("Loaded {} compounds.", vault.entries.len());
        }
    }

    // Input list (temporary seed)
    let file = File::open("compounds.txt").await?;
    let mut lines = BufReader::new(file).lines();

    let mut tasks = Vec::new();

    while let Some(line) = lines.next_line().await? {
        let name = line.trim().to_string();
        if name.is_empty() || seen.contains(&name) {
            continue;
        }

        let scraper_clone = Arc::clone(&scraper);

        tasks.push(tokio::spawn(async move {
            scraper_clone.scrape_chemical(name).await
        }));
    }

    for task in tasks {
        match task.await {
            Ok(Some(chem)) => {
                println!("Added: {}", chem.name);
                vault.entries.insert(chem.name.clone(), chem);
            }
            _ => println!("Skipped/Failed"),
        }
    }

    // Save
    let mut buf = Vec::new();
    vault.encode(&mut buf)?;
    std::fs::write("chemical_vault.bin.tmp", buf)?;
    std::fs::rename("chemical_vault.bin.tmp", "chemical_vault.bin")?;

    println!("Final count: {}", vault.entries.len());

    Ok(())
}
