use libhackrf::HackRf;
use num_complex::Complex;
use rustfft::{FftPlanner, Fft};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::any::Any;
use ndarray::{Array2, s};

const FREQS: [u64; 4] = [2_412_000_000, 2_432_000_000, 2_452_000_000, 2_472_000_000];
const WINDOW_SIZE: usize = 100; // Tamanho da janela temporal (Ex: 100 amostras)
const NUM_SLOTS: usize = 512;

lazy_static::lazy_static! {
    static ref FULL_SPECTRUM: Mutex<Vec<f32>> = Mutex::new(vec![0.0; NUM_SLOTS]);
    static ref CURRENT_STEP: Mutex<usize> = Mutex::new(0);
    static ref SKIP_COUNT: Mutex<usize> = Mutex::new(0);
    
    //JANELA DESLIZANTE (100 linhas x 512 colunas)
    static ref WINDOW_BUFFER: Mutex<Array2<f32>> = Mutex::new(Array2::zeros((WINDOW_SIZE, NUM_SLOTS)));
    static ref SAMPLES_COUNT: Mutex<usize> = Mutex::new(0);

    static ref FFT_PLAN: Arc<dyn Fft<f32>> = {
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(512)
    };
    static ref SOCKET: UdpSocket = {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect("127.0.0.1:5005").unwrap();
        s
    };
}

fn main() {
    let mut hackrf = HackRf::open().expect("HackRF não encontrado.");
    hackrf.set_sample_rate(20_000_000).unwrap();
    let _ = hackrf.set_amp_enable(false);
    let _ = hackrf.set_lna_gain(32);
    let _ = hackrf.set_rxvga_gain(40);

    println!("Janelas Deslizantes: A acumular {} amostras por slot...", WINDOW_SIZE);

    hackrf.start_rx(|device, samples: &[Complex<i8>], _user_data: &dyn Any| {
        let mut skip = SKIP_COUNT.lock().unwrap();
        if *skip < 10 { *skip += 1; return; }

        let mut buffer: Vec<Complex<f32>> = samples.iter()
            .take(512)
            .map(|c| Complex::new(c.re as f32 / 128.0, c.im as f32 / 128.0))
            .collect();

        if buffer.len() == 512 {
            FFT_PLAN.process(&mut buffer);
            let mut spectrum = FULL_SPECTRUM.lock().unwrap();
            let mut step = CURRENT_STEP.lock().unwrap();

            let offset = *step * 128;
            for (i, chunk) in buffer.chunks_exact(4).enumerate().take(128) {
                let p: f32 = chunk.iter().map(|c| (c.re*c.re + c.im*c.im).sqrt()).sum::<f32>() / 4.0;
                spectrum[offset + i] = p;
            }

            *step = (*step + 1) % 4;
            let _ = device.set_freq(FREQS[*step]);
            *skip = 0;

            if *step == 0 {
                let mut win = WINDOW_BUFFER.lock().unwrap();
                let mut count = SAMPLES_COUNT.lock().unwrap();

                // Lógica de Deslize (Shift)
                if *count < WINDOW_SIZE {
                    for s in 0..NUM_SLOTS {
                        win[[*count, s]] = spectrum[s];
                    }
                    *count += 1;
                } else {
                    // Move as linhas 1..100 para a posição 0..99
                    let mut temp_matrix = win.clone();
                    win.slice_mut(s![0..WINDOW_SIZE-1, ..])
                       .assign(&temp_matrix.slice(s![1..WINDOW_SIZE, ..]));
                    
                    // Inserir a nova linha na última posição
                    for s in 0..NUM_SLOTS {
                        win[[WINDOW_SIZE - 1, s]] = spectrum[s];
                    }
                    
                    // AQUI É ONDE VAMOS CALCULAR AS MÉTRICAS (PRÓXIMO PASSO)
                    // println!("Extração de features!");
                }

                // Enviar para o Python continuar a mostrar o gráfico em tempo real
                let bytes: Vec<u8> = spectrum.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect();
                let _ = SOCKET.send(&bytes);
            }
        }
    }, ()).unwrap();

    loop { std::thread::sleep(std::time::Duration::from_secs(1)); }
}