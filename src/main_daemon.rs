use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::net::UdpSocket;
use std::sync::Mutex;
use ndarray::{Array2, s};
use std::fs::OpenOptions;
use std::time::Duration;

// --- CONFIGURAÇÕES DA TESE ---
const START_FREQ: u64 = 2400_000_000;
const BIN_WIDTH: u64 = 156250; 
const NUM_SLOTS: usize = 512;
const WINDOW_SIZE: usize = 5000; // 5000 amostras (aprox. 1 minuto)
const SLIDING_STEP: usize = 500; // Desliza e extrai métricas a cada ~5 segundos 
const NUM_CHANNELS: usize = 14;
const SLOTS_PER_CHANNEL: usize = 36;

lazy_static::lazy_static! {
    static ref ROW_BUFFER: Mutex<Vec<f32>> = Mutex::new(vec![-100.0; NUM_SLOTS]);
    static ref WINDOW_BUFFER: Mutex<Array2<f32>> = Mutex::new(Array2::zeros((WINDOW_SIZE, NUM_SLOTS)));
    static ref FILLED_ROWS: Mutex<usize> = Mutex::new(0);
    static ref ROWS_SINCE_LAST: Mutex<usize> = Mutex::new(0);

    static ref SOCKET: UdpSocket = {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect("127.0.0.1:5005").unwrap();
        s
    };
}

fn main() {
    println!(">>> DAEMON LIVE: Sweep de Hardware + Extração de Features");
    
    let mut child = Command::new("hackrf_sweep")
        .arg("-f").arg("2400:2480")
        .arg("-w").arg(BIN_WIDTH.to_string())
        .arg("-l").arg("32")
        .arg("-g").arg("30")
        .arg("-a").arg("1")
        .stdout(Stdio::piped())
        .spawn()
        .expect("Erro ao iniciar hackrf_sweep");

    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        if let Ok(l) = line {
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() > 6 {
                let current_low_hz = parts[2].trim().parse::<u64>().unwrap_or(0);
                let start_slot = if current_low_hz > START_FREQ {
                    ((current_low_hz - START_FREQ) / BIN_WIDTH) as usize
                } else { 0 };

                let mut row = ROW_BUFFER.lock().unwrap();

                if start_slot == 0 && !row.iter().all(|&x| x == -100.0) {
                    let mut win = WINDOW_BUFFER.lock().unwrap();
                    let mut count = FILLED_ROWS.lock().unwrap();

                    if *count < WINDOW_SIZE {
                        for s in 0..NUM_SLOTS { win[[*count, s]] = row[s]; }
                        *count += 1;
                    } else {
                        let current_data = win.clone();
                        win.slice_mut(s![0..WINDOW_SIZE-1, ..])
                           .assign(&current_data.slice(s![1..WINDOW_SIZE, ..]));
                        for s in 0..NUM_SLOTS { win[[WINDOW_SIZE - 1, s]] = row[s]; }
                        
                        let mut steps = ROWS_SINCE_LAST.lock().unwrap();
                        *steps += 1;
                        if *steps >= SLIDING_STEP {
                            process_all_channels(&win, "wifi_normal_casa_validacao.csv");
                            *steps = 0;
                        }
                    }

                    let bytes: Vec<u8> = row.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect();
                    let _ = SOCKET.send(&bytes);
                }

                for (i, part) in parts.iter().skip(6).enumerate() {
                    let idx = start_slot + i;
                    if idx < NUM_SLOTS {
                        if let Ok(db) = part.trim().parse::<f32>() { row[idx] = db; }
                    }
                }
            }
        }
    }
}

fn process_all_channels(matrix: &Array2<f32>, output_path: &str) {
    let mut all_features = Vec::new();
    let mut has_any_activity = false;

    for c in 0..NUM_CHANNELS {
        let f_start = c * SLOTS_PER_CHANNEL;
        let f_end = (f_start + SLOTS_PER_CHANNEL).min(NUM_SLOTS);
        let roi = matrix.slice(s![.., f_start..f_end]);
        
        let metrics = calculate_roi_metrics(&roi.to_owned());
        
        if metrics.is_empty() {
            // Se o canal está morto, pomos zeros
            all_features.extend(vec![0.0; 14]);
        } else {
            all_features.extend(metrics);
            has_any_activity = true; // Pelo menos um canal tem "vida"
        }
    }

    // Só guardamos no CSV se a banda de 80MHz não estiver totalmente morta
    if has_any_activity {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(output_path) {
            let line = all_features.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            let _ = writeln!(file, "{}", line);
        }
    }
}

fn calculate_roi_metrics(roi: &Array2<f32>) -> Vec<f32> {
    let nt = roi.nrows() as f32; // Agora Nt = 1000
    let nf = roi.ncols() as f32; // Nf = 36

    // 1. Verificação de Atividade (Filtro de Silêncio Total)
    let max_power = roi.iter().fold(f32::MIN, |a, &b| a.max(b));
    
    // Se o canal estiver totalmente "morto" (abaixo de -70dB), ignoramos
    //if max_power < -70.0 {
    //    return Vec::new(); 
    //}

    // 2. Cálculo das 14 Métricas (Se houver atividade)
    let mut f = Vec::new();
    
    // Média Bidimensional (µ)
    f.push(roi.mean().unwrap_or(0.0));

    // Máximo Médio e Variância
    let mut max_sum = 0.0;
    let mut row_variances = Vec::new();
    for row in roi.rows() {
        let r_vec = row.to_vec();
        max_sum += r_vec.iter().fold(f32::MIN, |a, &b| a.max(b));
        let r_mean = r_vec.iter().sum::<f32>() / nf;
        let r_var = r_vec.iter().map(|x| (x - r_mean).powi(2)).sum::<f32>() / nf;
        row_variances.push(r_var);
    }
    f.push(max_sum / nt);
    f.push(row_variances.iter().sum::<f32>() / nt);

    // Percentis (P50, P75, P90, P95, P99)
    let mut flat: Vec<f32> = roi.iter().cloned().collect();
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = flat.len() as f32;
    f.push(flat[(len * 0.50) as usize]);
    f.push(flat[(len * 0.75) as usize]);
    f.push(flat[(len * 0.90) as usize]);
    f.push(flat[(len * 0.95) as usize]);
    f.push(flat[(len * 0.99) as usize]);

    // Markov e Entropia (O "Ritmo")
    let sample_threshold = -70.0; 
    let at: Vec<i32> = roi.rows().into_iter()
        .map(|r| if r.iter().any(|&x| x > sample_threshold) { 1 } else { 0 })
        .collect();

    let active_samples = at.iter().filter(|&&x| x == 1).count() as f32;
    f.push(active_samples / nt); // Taxa de Ocupação

    let (mut n11, mut n10, mut n01, mut n00) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..at.len()-1 {
        match (at[i], at[i+1]) {
            (1, 1) => n11 += 1.0, (1, 0) => n10 += 1.0,
            (0, 1) => n01 += 1.0, (0, 0) => n00 += 1.0,
            _ => (),
        }
    }
    
    let total_act = active_samples;
    let total_sil = nt - active_samples;

    f.push(if total_act > 0.0 { n11 / total_act } else { 0.0 }); // P(1|1)
    f.push(if total_act > 0.0 { n10 / total_act } else { 0.0 }); // P(0|1)
    f.push(if total_sil > 0.0 { n01 / total_sil } else { 0.0 }); // P(1|0)
    f.push(if total_sil > 0.0 { n00 / total_sil } else { 0.0 }); // P(0|0)
    
    let p1 = active_samples / nt;
    let p0 = 1.0 - p1;
    f.push(if p1 > 0.0 && p0 > 0.0 { -(p1*p1.log2() + p0*p0.log2()) } else { 0.0 });

    f
}