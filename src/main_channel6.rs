use libhackrf::HackRf;
use num_complex::Complex;
use rustfft::{FftPlanner, Fft};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::any::Any;

lazy_static::lazy_static! {
    // Vetor global de 512 slots (os "canais todos")
    static ref FULL_SPECTRUM: Mutex<Vec<f32>> = Mutex::new(vec![0.0; 512]);
    static ref CURRENT_STEP: Mutex<usize> = Mutex::new(0);
    static ref FFT_PLAN: Arc<dyn Fft<f32>> = {
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(512) // FFT menor para ser mais rápido no sweep
    };
    static ref SOCKET: UdpSocket = {
        let s = UdpSocket::bind("127.0.0.1:0").unwrap();
        s.connect("127.0.0.1:5005").unwrap();
        s
    };
}

fn main() {
    let mut hackrf = HackRf::open().expect("HackRF não encontrado.");
    
    // FIXO no Canal 6 (2.437 GHz)
    hackrf.set_freq(2_437_000_000).unwrap(); 
    hackrf.set_sample_rate(20_000_000).unwrap();
    
    let _ = hackrf.set_amp_enable(false);
    let _ = hackrf.set_lna_gain(32);
    let _ = hackrf.set_rxvga_gain(40);

    println!("MODO TESTE: Fixo no Canal 6 (2.437 GHz). Liga o vídeo!");

    hackrf.start_rx(|_device, samples: &[Complex<i8>], _user_data: &dyn Any| {
        let mut buffer: Vec<Complex<f32>> = samples.iter()
            .take(2048)
            .map(|c| Complex::new(c.re as f32 / 128.0, c.im as f32 / 128.0))
            .collect();

        if buffer.len() == 2048 {
            FFT_PLAN.process(&mut buffer);

            // Enviamos os 512 slots (desta vez todos da mesma zona)
            let mut slots_512 = Vec::with_capacity(512);
            for chunk in buffer.chunks_exact(4) {
                let p: f32 = chunk.iter().map(|c| (c.re*c.re + c.im*c.im).sqrt()).sum::<f32>() / 4.0;
                slots_512.push(p);
            }

            let bytes: Vec<u8> = slots_512.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect();
            let _ = SOCKET.send(&bytes);
        }
    }, ()).unwrap();

    loop { std::thread::sleep(std::time::Duration::from_secs(1)); }
}