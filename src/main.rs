use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::net::UdpSocket;
use std::sync::Mutex;
use ndarray::{Array2, s};
use std::fs::OpenOptions;


// --- CONFIGURAÇÕES DA TESE (80 MHz e Janelas) ---
const START_FREQ: u64 = 2400_000_000;
const BIN_WIDTH: u64 = 156250; // 80MHz / 512 slots
const NUM_SLOTS: usize = 512;
const WINDOW_SIZE: usize = 100; // O tamanho da tua Janela Deslizante

// NOVO: Define a cada quantas novas linhas/varrimentos queres extrair as métricas (Os teus 15 segundos)
const SLIDING_STEP: usize = 15; 

lazy_static::lazy_static! {
    // 1. Buffer para a linha atual que está a ser "cosida"
    static ref ROW_BUFFER: Mutex<Vec<f32>> = Mutex::new(vec![-100.0; NUM_SLOTS]);
    
    // 2. Buffer para a Janela Deslizante (As últimas 100 linhas para as métricas)
    static ref WINDOW_BUFFER: Mutex<Array2<f32>> = Mutex::new(Array2::zeros((WINDOW_SIZE, NUM_SLOTS)));
    static ref FILLED_ROWS: Mutex<usize> = Mutex::new(0);
    static ref ROWS_SINCE_LAST_EXTRACTION: Mutex<usize> = Mutex::new(0);

    static ref SOCKET: UdpSocket = {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect("127.0.0.1:5005").unwrap();
        s
    };
}


// =====================================================================
// FUNÇÕES DE EXTRAÇÃO DE FEATURES
// =====================================================================

fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n == 0.0 { return 0.0; }
    let (mut sa, mut sb, mut sab, mut sa2, mut sb2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..a.len() {
        sa += a[i]; sb += b[i]; sab += a[i]*b[i]; sa2 += a[i]*a[i]; sb2 += b[i]*b[i];
    }
    let num = n * sab - sa * sb;
    let den = ((n * sa2 - sa * sa) * (n * sb2 - sb * sb)).sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

fn extract_features_for_channel(matrix: &Array2<f32>, channel_num: usize, f_start: usize, f_end: usize) {
    let roi = matrix.slice(s![0..WINDOW_SIZE, f_start..f_end]);
    let num_slots_roi = roi.ncols(); 
    
    let mut row_features = Vec::new();
    row_features.push(format!("Canal_{}", channel_num));

    // 2. Extrair Métricas Temporais para cada slot do retângulo
    for s_idx in 0..num_slots_roi {
        // AQUI ESTÁ O TEMPO REAL (Cronológico)
        let col_time = roi.column(s_idx).to_vec();
        
        // --- 1º PASSO: CÁLCULO DAS ESTATÍSTICAS (Com ordenação) ---
        let mut col_sorted = col_time.clone(); 
        col_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let len = col_sorted.len() as f32;
        let mean = col_sorted.iter().sum::<f32>() / len;
        let max = *col_sorted.last().unwrap_or(&0.0);
        let min = *col_sorted.first().unwrap_or(&0.0);
        let median = col_sorted[col_sorted.len() / 2];
        let std_dev = (col_sorted.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len).sqrt();
        
        let p75 = col_sorted[((len * 0.75) as usize).min(col_sorted.len() - 1)];
        let p90 = col_sorted[((len * 0.90) as usize).min(col_sorted.len() - 1)];
        let p95 = col_sorted[((len * 0.95) as usize).min(col_sorted.len() - 1)];
        let p99 = col_sorted[((len * 0.99) as usize).min(col_sorted.len() - 1)];
        
        // --- 2º PASSO: THRESHOLD DINÂMICO ---
        // O silêncio é o ruído de fundo (min) mais uma margem de segurança (ex: 7.0 dB).
        // Qualquer transmissão real vai saltar 15, 20 ou 30 dB acima do ruído.
        let silence_threshold = min + 7.0; 

        // --- 3º PASSO: CÁLCULO DOS SILÊNCIOS NO TEMPO (Usando o threshold dinâmico) ---
        let mut silence_count = 0;
        let mut current_silence_run = 0;
        let mut max_silence_run = 0;

        for &val in &col_time {
            if val <= silence_threshold {
                silence_count += 1;
                current_silence_run += 1;
                if current_silence_run > max_silence_run {
                    max_silence_run = current_silence_run;
                }
            } else {
                current_silence_run = 0; 
            }
        }
        
        let silence_ratio = silence_count as f32 / len;
        let max_silence = max_silence_run as f32;

        // --- 4º PASSO: Juntar tudo ---
        for metric in[mean, max, min, median, std_dev, p75, p90, p95, p99, silence_ratio, max_silence] { 
            row_features.push(metric.to_string());
        }
    }

    // 3. Correlação de Pearson Espetral (adjacente)
    for s_idx in 0..(num_slots_roi - 1) {
        let corr = pearson_correlation(&roi.column(s_idx).to_vec(), &roi.column(s_idx+1).to_vec());
        row_features.push(corr.to_string());
    }

    // 4. Gravar a linha final no ficheiro CSV
    if let Ok(file) = OpenOptions::new().create(true).append(true).open("features_dataset.csv") {
        let mut wtr = csv::Writer::from_writer(file);
        let _ = wtr.write_record(&row_features);
    }
}

// =====================================================================
// MAIN
// =====================================================================

fn main() {
    println!(">>> Motor Rust: Sweep + Janela Deslizante ({} amostras)...", WINDOW_SIZE);
    println!(">>> Extração de features a cada {} instâncias.", SLIDING_STEP);

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

                // Quando terminamos uma linha inteira e voltamos ao início da banda
                if start_slot == 0 && !row.iter().all(|&x| x == -100.0) {
                    
                    let mut win = WINDOW_BUFFER.lock().unwrap();
                    let mut count = FILLED_ROWS.lock().unwrap();

                    // FASE 1: Encher a matriz as primeiras 100 vezes
                    if *count < WINDOW_SIZE {
                        for s in 0..NUM_SLOTS { win[[*count, s]] = row[s]; }
                        *count += 1;
                    } 
                    // FASE 2: Matriz cheia, começar a deslizar e a extrair métricas
                    else {
                        // Deslizar as linhas antigas para cima
                        let current_data = win.clone();
                        win.slice_mut(s![0..WINDOW_SIZE-1, ..])
                           .assign(&current_data.slice(s![1..WINDOW_SIZE, ..]));
                        
                        // Meter a nova linha na última posição
                        for s in 0..NUM_SLOTS { win[[WINDOW_SIZE - 1, s]] = row[s]; }
                        
                        // --- NOVA LÓGICA DE EXTRAÇÃO COM SLIDING STEP ---
                        let mut steps = ROWS_SINCE_LAST_EXTRACTION.lock().unwrap();
                        *steps += 1; // Entrou 1 linha nova

                        // Atingimos os 15 saltos/linhas novas? Então calculamos as métricas!
                        if *steps >= SLIDING_STEP {
                            // Iterar pelos 13 canais Wi-Fi Europeus
                            for canal in 1..=13 {
                                // Exemplo matemático: Canal 1 = 2412 MHz
                                let f_centro = 2407 + 5 * canal; 
                                
                                // Converter MHz para índice de slot no nosso buffer de 512 slots
                                let diff_hz = (f_centro * 1_000_000) - START_FREQ;
                                let slot_central = (diff_hz / BIN_WIDTH) as usize;
                                
                                // Criar as paredes esquerda e direita do "Retângulo" (18 slots para cada lado = 36 slots)
                                let f_start = slot_central.saturating_sub(18);
                                let f_end = (slot_central + 18).min(NUM_SLOTS);
                                
                                // Chama a função que recorta o retângulo e guarda as estatísticas no CSV!
                                extract_features_for_channel(&win, canal as usize, f_start, f_end);
                            }
                            // Extração feita, volta a colocar o contador a zero!
                            *steps = 0; 
                        }
                    }

                    // Enviar os dados por UDP para o teu visualizador em Python manter-se fluído
                    let bytes: Vec<u8> = row.iter()
                        .take(NUM_SLOTS)
                        .flat_map(|&f| f.to_le_bytes().to_vec())
                        .collect();
                    let _ = SOCKET.send(&bytes);
                }

                // Preencher o buffer com as novas varreduras
                for (i, part) in parts.iter().skip(6).enumerate() {
                    let idx = start_slot + i;
                    if idx < NUM_SLOTS {
                        if let Ok(db) = part.trim().parse::<f32>() {
                            row[idx] = db;
                        }
                    }
                }
            }
        }
    }
}