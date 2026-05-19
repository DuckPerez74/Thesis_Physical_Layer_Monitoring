use std::io::{BufRead, BufReader, Write};
use std::fs::{File, OpenOptions};
use ndarray::{Array2, s};
use std::any::Any;

// --- CONFIGURAÇÕES TÉCNICAS ---
const WINDOW_SIZE: usize = 5000; // 50seg 
const NUM_SLOTS: usize = 512;   
const NUM_CHANNELS: usize = 14; 
const SLOTS_PER_CHANNEL: usize = 36;
const SLIDING_STEP: usize = 500; // Extrair métricas a cada ~5 segundos (para termos uma boa quantidade de amostras)

fn main() {
    let input_path = "raw_attack_lvl4_staticstic.csv"; 
    let output_path = "features_attack_lvl4_staticstic.csv";

    println!(">>> Iniciando Processamento Offline: {}", input_path);
    println!(">>> Janela: {} | Step: {}", WINDOW_SIZE, SLIDING_STEP);

    let file = File::open(input_path).expect("Ficheiro não encontrado!");
    let reader = BufReader::new(file);

    // Inicialização correta da matriz
    let mut window_buffer = Array2::<f32>::zeros((WINDOW_SIZE, NUM_SLOTS));
    let mut rows_filled = 0;

    for (line_idx, line) in reader.lines().enumerate() {
        if let Ok(l) = line {
            let values: Vec<f32> = l.split(',')
                .map(|s| s.parse().unwrap_or(-100.0))
                .collect();

            if values.len() >= NUM_SLOTS {
                if rows_filled < WINDOW_SIZE {
                    // 1. Fase de enchimento inicial
                    for s in 0..NUM_SLOTS {
                        window_buffer[[rows_filled, s]] = values[s];
                    }
                    rows_filled += 1;
                } else {
                    // 2. Janela Cheia: Lógica de Deslize (Sliding)
                    // Criamos uma cópia das linhas 1 até 999
                    let shifted_data = window_buffer.slice(s![1..WINDOW_SIZE, ..]).to_owned();
                    // Colocamos essas linhas nas posições 0 até 998
                    window_buffer.slice_mut(s![0..WINDOW_SIZE-1, ..]).assign(&shifted_data);
                    // Inserimos a linha nova na última posição (999)
                    for s in 0..NUM_SLOTS {
                        window_buffer[[WINDOW_SIZE - 1, s]] = values[s];
                    }

                    // 3. Só processamos features após o enchimento e respeitando o STEP
                    if line_idx % SLIDING_STEP == 0 {
                        process_all_channels(&window_buffer, output_path);
                    }
                }
            }
        }
        if line_idx % 10000 == 0 { println!("Lidas {} linhas...", line_idx); }
    }
    println!(">>> CONCLUÍDO! Dataset gerado.");
}

fn process_all_channels(matrix: &Array2<f32>, output_path: &str) {
    let mut all_features = Vec::new();
    let mut has_activity = false;

    for c in 0..NUM_CHANNELS {
        let f_start = c * SLOTS_PER_CHANNEL;
        let f_end = (f_start + SLOTS_PER_CHANNEL).min(NUM_SLOTS);
        
        let roi = matrix.slice(s![.., f_start..f_end]);
        let metrics = calculate_roi_metrics(&roi.to_owned());
        
        if metrics.is_empty() {
            all_features.extend(vec![0.0; 14]);
        } else {
            all_features.extend(metrics);
            has_activity = true;
        }
    }

    // Gravação condicional (Gating de Silêncio)
    if has_activity {
        let mut file = OpenOptions::new().create(true).append(true).open(output_path).unwrap();
        let line = all_features.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        writeln!(file, "{}", line).unwrap();
    }
}

fn calculate_roi_metrics(roi: &Array2<f32>) -> Vec<f32> {
    let max_val = roi.iter().fold(f32::MIN, |a, &b| a.max(b));
    
    // GATING DE SILÊNCIO: Se a janela está morta, retorna vazio
    //if max_val < -70.0 { return Vec::new(); }

    let nt = roi.nrows() as f32;
    let nf = roi.ncols() as f32;
    let mut f = Vec::new();

    // Média 2D
    f.push(roi.mean().unwrap_or(0.0));

    // Máximo Médio e Variância
    let mut max_sum = 0.0;
    let mut row_vars = Vec::new();
    for row in roi.rows() {
        let r_vec = row.to_vec();
        max_sum += r_vec.iter().fold(f32::MIN, |a, &b| a.max(b));
        let r_mean = r_vec.iter().sum::<f32>() / nf;
        let r_var = r_vec.iter().map(|x| (x - r_mean).powi(2)).sum::<f32>() / nf;
        row_vars.push(r_var);
    }
    f.push(max_sum / nt);
    f.push(row_vars.iter().sum::<f32>() / nt);

    // Percentis
    let mut flat: Vec<f32> = roi.iter().cloned().collect();
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = flat.len() as f32;
    f.push(flat[(len * 0.50) as usize]);
    f.push(flat[(len * 0.75) as usize]);
    f.push(flat[(len * 0.90) as usize]);
    f.push(flat[(len * 0.95) as usize]);
    f.push(flat[(len * 0.99) as usize]);

    // Ocupação e Markov
    let thresh = -75.0;
    let act_count = flat.iter().filter(|&&x| x > thresh).count();
    f.push(act_count as f32 / len);

    let at: Vec<i32> = roi.rows().into_iter()
        .map(|r| if r.iter().any(|&x| x > thresh) { 1 } else { 0 })
        .collect();

    let (mut n11, mut n10, mut n01, mut n00) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..at.len()-1 {
        match (at[i], at[i+1]) {
            (1, 1) => n11 += 1.0, (1, 0) => n10 += 1.0,
            (0, 1) => n01 += 1.0, (0, 0) => n00 += 1.0,
            _ => (),
        }
    }
    let total_act = at.iter().filter(|&&x| x == 1).count() as f32;
    let total_sil = nt - total_act;

    f.push(if total_act > 0.0 { n11 / total_act } else { 0.0 }); 
    f.push(if total_act > 0.0 { n10 / total_act } else { 0.0 });
    f.push(if total_sil > 0.0 { n01 / total_sil } else { 0.0 }); 
    f.push(if total_sil > 0.0 { n00 / total_sil } else { 0.0 }); 
    
    let p1 = total_act / nt;
    let p0 = 1.0 - p1;
    f.push(if p1 > 0.0 && p0 > 0.0 { -(p1*p1.log2() + p0*p0.log2()) } else { 0.0 });

    f
}