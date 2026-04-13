use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::net::UdpSocket;
use std::sync::Mutex;
use ndarray::{Array2, s};

// --- CONFIGURAÇÕES DA TESE (80 MHz e Janelas) ---
const START_FREQ: u64 = 2400_000_000;
const BIN_WIDTH: u64 = 156250; // 80MHz / 512 slots
const NUM_SLOTS: usize = 512;
const WINDOW_SIZE: usize = 100; // O tamanho da tua Janela Deslizante

lazy_static::lazy_static! {
    // 1. Buffer para a linha atual que está a ser "cosida"
    static ref ROW_BUFFER: Mutex<Vec<f32>> = Mutex::new(vec![-100.0; NUM_SLOTS]);
    
    // 2. Buffer para a Janela Deslizante (As últimas 100 linhas para as métricas)
    static ref WINDOW_BUFFER: Mutex<Array2<f32>> = Mutex::new(Array2::zeros((WINDOW_SIZE, NUM_SLOTS)));
    static ref FILLED_ROWS: Mutex<usize> = Mutex::new(0);

    static ref SOCKET: UdpSocket = {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect("127.0.0.1:5005").unwrap();
        s
    };
}

fn main() {
    println!(">>> Motor Rust: Sweep de Hardware + Janela Deslizante ({} amostras)...", WINDOW_SIZE);

    // Chamamos o hackrf_sweep com os teus parâmetros de ganho do GitHub
    let mut child = Command::new("hackrf_sweep")
        .arg("-f").arg("2400:2480")
        .arg("-w").arg(BIN_WIDTH.to_string())
        .arg("-l").arg("32") // Ganho LNA do teu GitHub
        .arg("-g").arg("30") // Ganho VGA do teu GitHub
        .arg("-a").arg("1")  // Amplificador ligado para detetar vizinhos
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
                
                // MAPEAMENTO DE FREQUÊNCIA (A solução para as fendas)
                let start_slot = if current_low_hz > START_FREQ {
                    ((current_low_hz - START_FREQ) / BIN_WIDTH) as usize
                } else { 0 };

                let mut row = ROW_BUFFER.lock().unwrap();

                // Se voltámos ao início, processamos a Janela Deslizante e enviamos ao Python
                if start_slot == 0 && !row.iter().all(|&x| x == -100.0) {
                    
                    // --- LÓGICA DA JANELA DESLIZANTE ---
                    let mut win = WINDOW_BUFFER.lock().unwrap();
                    let mut count = FILLED_ROWS.lock().unwrap();

                    if *count < WINDOW_SIZE {
                        // Fase inicial: encher a matriz
                        for s in 0..NUM_SLOTS { win[[*count, s]] = row[s]; }
                        *count += 1;
                    } else {
                        // Janela cheia: DESLIZAR (Shift)
                        let current_data = win.clone();
                        win.slice_mut(s![0..WINDOW_SIZE-1, ..])
                           .assign(&current_data.slice(s![1..WINDOW_SIZE, ..]));
                        
                        // Inserir a nova varredura completa na última posição
                        for s in 0..NUM_SLOTS { win[[WINDOW_SIZE - 1, s]] = row[s]; }
                        
                        // AQUI: Já podes chamar a extração de métricas sobre 'win'
                        // save_metrics_to_csv(&win); 
                    }

                    // Enviar a linha para o Python (Gráfico)
                    let bytes: Vec<u8> = row.iter()
                        .take(NUM_SLOTS)
                        .flat_map(|&f| f.to_le_bytes().to_vec())
                        .collect();
                    let _ = SOCKET.send(&bytes);
                }

                // Preencher o ROW_BUFFER com os dados do novo segmento
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