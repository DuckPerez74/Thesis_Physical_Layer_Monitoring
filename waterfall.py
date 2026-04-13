import socket
import numpy as np
import matplotlib.pyplot as plt

# Configurações
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
HISTORY_LEN = 150 

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))

data_matrix = None
im = None

print(">>> Python: Visualização do Espectro Sincronizado.")

try:
    while True:
        packet, addr = sock.recvfrom(8192)
        if len(packet) > 0:
            # 1. Converter amostras (vêm do Rust)
            new_row = np.frombuffer(packet, dtype=np.float32).copy()
            num_bins = len(new_row)

            # --- 2. ADAPTAÇÃO AUTOMÁTICA AO TAMANHO ---
            if data_matrix is None or data_matrix.shape[1] != num_bins:
                print(f"Detectados {num_bins} slots. Ajustando matriz...")
                data_matrix = np.zeros((HISTORY_LEN, num_bins)) - 100
                if im is not None:
                    im.remove()
                # Usamos interpolation='bilinear' para o aspeto fluido e profissional
                im = ax.imshow(data_matrix, aspect='auto', cmap='magma', 
                               extent=[2400, 2480, HISTORY_LEN, 0], interpolation='bilinear')
                plt.title("Espectrograma Sincronizado $P(f,t)$ - 80MHz")
                plt.xlabel("Frequência (MHz)")
                plt.ylabel("Tempo (Janelas)")

            # --- 3. LIMPEZA LEVE DO DC SPIKE CENTRAL ---
            # Como o Rust já manda tudo ok, só suavizamos o ponto central exato
            mid = num_bins // 2
            new_row[mid-1:mid+2] = np.median(new_row)

            # --- 4. ATUALIZAR MATRIZ COM PERSISTÊNCIA REATIVA ---
            # 0.5 é um bom equilíbrio: mantém o sinal sólido mas sem arrastamento
            decay = 0.5
            data_matrix = np.roll(data_matrix, 1, axis=0)
            data_matrix[0, :] = (data_matrix[1, :] * decay) + (new_row * (1 - decay))
            
            # --- 5. CONTRASTE DINÂMICO (Cores Vivas) ---
            im.set_array(data_matrix)
            # Definimos o 'preto' no percentil 35 e o 'amarelo' nos picos reais
            floor = np.percentile(data_matrix, 35)
            peak = np.percentile(data_matrix, 99.9)
            
            # Forçamos uma escala que destaca o Wi-Fi sem saturar o ruído
            im.set_clim(vmin=floor, vmax=max(floor + 10, peak))
            
            plt.pause(0.001)

except KeyboardInterrupt:
    print("\nParado pelo utilizador.")
finally:
    sock.close()