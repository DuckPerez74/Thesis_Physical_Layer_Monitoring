use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::fs::OpenOptions;
use std::io::Write;

fn main() {
    let mut full_row: Vec<f32> = Vec::with_capacity(512);
    let csv_path = "raw_attack_lvl4_staticstic.csv";

    println!(">>> INICIANDO CAPTURA DOS ATAQUES (FORMATO RAW)");
    println!(">>> Gravando dados em: {}", csv_path);

    let mut child = Command::new("hackrf_sweep")
        .arg("-f").arg("2400:2480")
        .arg("-w").arg("156250") // 512 slots
        .arg("-l").arg("32")
        .arg("-g").arg("30")
        .arg("-a").arg("1")
        .stdout(Stdio::piped())
        .spawn()
        .expect("Erro ao iniciar hackrf_sweep");
    
    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    // Abrir o ficheiro uma vez para append
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)
        .expect("Erro ao abrir ficheiro CSV");

    for line in reader.lines() {
        if let Ok(l) = line {
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() > 6 {
                if parts[2].trim() == "2400000000" {
                    if full_row.len() >= 500 {
                        // Transformar os 512 números numa linha de texto CSV
                        let csv_line = full_row.iter()
                            .take(512)
                            .map(|f| f.to_string())
                            .collect::<Vec<String>>()
                            .join(",");
                        
                        writeln!(file, "{}", csv_line).unwrap();
                    }
                    full_row.clear();
                }

                for i in 6..parts.len() {
                    if let Ok(db) = parts[i].trim().parse::<f32>() {
                        full_row.push(db);
                    }
                }
            }
        }
    }
}