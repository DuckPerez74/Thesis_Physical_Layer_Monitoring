use libhackrf::HackRf;
use num_complex::Complex;
use rustfft::{FftPlanner, Fft};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::any::Any;
use ndarray::{Array2, s}; // Importamos ndarray para a matriz da janela

const FREQS: [u64; 4] = [2_412_000_000, 2_432_000_000, 2_452_000_000, 2_472_000_000];
const FFT_LEN: usize = 1024;
const NUM_SLOTS: usize = 512;
const WINDOW_SIZE: usize = 100; // Tamanho da janela (ex: 100 linhas de tempo)

lazy_static::lazy_static! {
    static ref FULL_SPECTRUM: Mutex<Vec<f32>> = Mutex::new(vec![0.0; NUM_SLOTS]);
    static ref CURRENT_STEP: Mutex<usize> = Mutex::new(0);
    static ref SKIP_COUNT: Mutex<usize> = Mutex::new(0);

    // --- IMPLEMENTAÇÃO DA JANELA DESLIZANTE ---
    // Matriz 2D: 100 linhas (tempo) x 512 colunas (frequência)
    static ref WINDOW_BUFFER: Mutex<Array2<f32>> = Mutex::new(Array2::zeros((WINDOW_SIZE, NUM_SLOTS)));
    static ref FILLED_SAMPLES: Mutex<usize> = Mutex::new(0);
    // ------------------------------------------

    static ref FFT_PLAN: Arc<dyn Fft<f32>> = {
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(FFT_LEN)
    };
    static ref SOCKET: UdpSocket = {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect("127.0.0.1:5005").unwrap();
        s
    };
}

fn main() {
    let mut hackrf = HackRf::open().expect("HackRF não encontrado.");
    hackrf.set_sample_rate(10_000_000).unwrap(); 
    let _ = hackrf.set_amp_enable(false);
    let _ = hackrf.set_lna_gain(32);
    let _ = hackrf.set_rxvga_gain(40);

    println!("Modo SWEEP + JANELA DESLIZANTE ({} linhas).", WINDOW_SIZE);

    hackrf.start_rx(|device, samples: &[Complex<i8>], _user_data: &dyn Any| {
        let mut skip = SKIP_COUNT.lock().unwrap();
        if *skip < 5 { *skip += 1; return; }

        let mut buffer: Vec<Complex<f32>> = samples.iter()
            .take(FFT_LEN)
            .map(|c| Complex::new(c.re as f32 / 128.0, c.im as f32 / 128.0))
            .collect();

        if buffer.len() == FFT_LEN {
            FFT_PLAN.process(&mut buffer);

            let mut spectrum = FULL_SPECTRUM.lock().unwrap();
            let mut step = CURRENT_STEP.lock().unwrap();

            let start_bin = (FFT_LEN / 2) - 64;
            let offset = *step * 128;
            
            for i in 0..128 {
                let original_idx = (start_bin + i + FFT_LEN/2) % FFT_LEN;
                let spec = buffer[original_idx];
                spectrum[offset + i] = (spec.re.powi(2) + spec.im.powi(2)).sqrt();
            }

            *step = (*step + 1) % 4;
            let _ = device.set_freq(FREQS[*step]);
            *skip = 0;

            if *step == 0 {
                let mut win = WINDOW_BUFFER.lock().unwrap();
                let mut samples_count = FILLED_SAMPLES.lock().unwrap();

                if *samples_count < WINDOW_SIZE {
                    // Caso 1: Ainda a encher a primeira janela (fase inicial)
                    for s in 0..NUM_SLOTS {
                        win[[*samples_count, s]] = spectrum[s];
                    }
                    *samples_count += 1;
                } else {
                    // Caso 2: JANELA DESLIZANTE REAL (Slide)
                    // 1. Criamos uma cópia temporária para o shift
                    let win_copy = win.clone();
                    
                    // 2. Deslizamos as linhas: a 1 passa a ser a 0, a 2 passa a ser a 1...
                    // O slice s![1..WINDOW_SIZE, ..] pega nas linhas da 1 à 99
                    win.slice_mut(s![0..WINDOW_SIZE-1, ..])
                       .assign(&win_copy.slice(s![1..WINDOW_SIZE, ..]));
                    
                    // 3. Inserimos a nova amostra (a mais recente) na última linha (posição 99)
                    for s in 0..NUM_SLOTS {
                        win[[WINDOW_SIZE - 1, s]] = spectrum[s];
                    }
                }
                // ----------------------------------------------------

                let bytes: Vec<u8> = spectrum.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect();
                let _ = SOCKET.send(&bytes);
            }
        }
    }, ()).unwrap();

    loop { std::thread::sleep(std::time::Duration::from_millis(10)); }
}