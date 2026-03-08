use prost::Message;
use reqwest::Client;
use std::fs::File as StdFile;
use std::io::{Read, Write};
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
            client: Client::new(),
            semaphore: Arc::new(Semaphore::new(5)), // High-concurrency throttle
        }
    }

    async fn scrape_chemical(&self, name: String) -> Result<skynet::Chemical, Box<dyn std::error::Error + Send + Sync>> {
        let _permit = self.semaphore.acquire().await?;
        
        let url = format!(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON",
            name
        );

        let resp = self.client.get(url).send().await?.json::<serde_json::Value>().await?;
        
        let prop = &resp["PropertyTable"]["Properties"][0];
        let cid = prop["CID"].as_i64().ok_or("CID not found")? as i32;
        let smiles = prop["CanonicalSMILES"].as_str().unwrap_or("").to_string();

        Ok(skynet::Chemical {
            name: name.to_lowercase(),
            pubchem_cid: cid,
            smiles,
            ..Default::default()
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scraper = Arc::new(TitanScraper::new());
    let mut vault = skynet::Vault::default();

    // 1. Load existing vault
    if let Ok(mut file) = StdFile::open("chemical_vault.bin") {
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if let Ok(decoded) = skynet::Vault::decode(&buf[..]) {
            vault = decoded;
            println!("Loaded vault: {} compounds found.", vault.entries.len());
        }
    }

    // 2. Open compounds.txt
    let file = File::open("compounds.txt").await?;
    let mut lines = BufReader::new(file).lines();
    let mut tasks = Vec::new();

    while let Some(line) = lines.next_line().await? {
        let chemical_name = line.trim().to_string();
        if chemical_name.is_empty() { continue; }

        let key = chemical_name.to_lowercase();

        // --- THE SKIP FUNCTION ---
        if vault.entries.contains_key(&key) {
            println!("Skipping [{}]: Already present in Vault.", chemical_name);
            continue;
        }

        let s_clone = Arc::clone(&scraper);
        tasks.push(tokio::spawn(async move {
            s_clone.scrape_chemical(chemical_name).await
        }));
    }

    // 3. Process new additions
    for task in tasks {
        match task.await {
            Ok(Ok(chem)) => {
                println!("New Induction: {} (CID: {})", chem.name, chem.pubchem_cid);
                vault.entries.insert(chem.name.clone(), chem);
            }
            Ok(Err(e)) => eprintln!("Scrape failed: {}", e),
            Err(e) => eprintln!("Task join error: {}", e),
        }
    }

    // 4. Final Atomic Commit
    let mut buf = Vec::new();
    vault.encode(&mut buf)?;
    std::fs::write("chemical_vault.bin.tmp", buf)?;
    std::fs::rename("chemical_vault.bin.tmp", "chemical_vault.bin")?;

    println!("Process complete. Total compounds in Vault: {}", vault.entries.len());
    Ok(())
}