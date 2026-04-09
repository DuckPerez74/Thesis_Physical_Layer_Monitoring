import socket
import numpy as np
import matplotlib.pyplot as plt

# Configuração do Socket
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
NUM_SLOTS = 512
HISTORY_LEN = 150 

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))


# Definições da gama física
f_start = 2400  # MHz
f_end = 2480    # MHz

# Preparar o gráfico
plt.ion()
data_matrix = np.zeros((HISTORY_LEN, NUM_SLOTS))
fig, ax = plt.subplots(figsize=(12, 7))

# O parâmetro 'extent' define as etiquetas dos eixos: [esquerda, direita, baixo, cima]
# Usamos f_start e f_end para o eixo X
im = ax.imshow(data_matrix, aspect='auto', cmap='magma', animated=True, 
               interpolation='none', extent=[f_start, f_end, HISTORY_LEN, 0])

plt.title("Espectrograma em Tempo Real - Banda Completa Wi-Fi 2.4 GHz")
plt.xlabel("Frequência (MHz)")
plt.ylabel("Tempo (Janelas)")
plt.colorbar(im, label="Intensidade (dB)")

print("A aguardar dados... Certifica-te que o Rust está a correr!")

try:
    while True:
        packet, addr = sock.recvfrom(4096)
        
        # O bloco abaixo tem de estar identado (alinhado à direita)
        if len(packet) == NUM_SLOTS * 4:
            # 1. Converter bytes para float e criar cópia editável
            new_row = np.frombuffer(packet, dtype=np.float32).copy()
            
            # 2. ELIMINAR O DC SPIKE (Ruído central do hardware)
            center = NUM_SLOTS // 2
            new_row[center-2 : center+2] = (new_row[center-3] + new_row[center+3]) / 2

            # 3. CONVERTER PARA ESCALA LOG (dB)
            new_row_log = 10 * np.log10(new_row + 1e-6)

            # 4. Atualizar Matriz (Efeito de cascata)
            data_matrix = np.roll(data_matrix, 1, axis=0)
            data_matrix[0, :] = new_row_log
            
            # 5. Atualizar Gráfico com Contraste Automático
            im.set_array(data_matrix)
            
            # Ajustar os limites de cor com base nos percentis (brilho dinâmico)
            low = np.percentile(data_matrix, 40) # Aumentar para 40 limpa o ruído de fundo
            high = max(low + 15, np.percentile(data_matrix, 99.9))
            
            im.set_clim(vmin=low, vmax=high)

            
            plt.pause(0.001)
            
except KeyboardInterrupt:
    print("\nCaptura terminada.")
finally:
    sock.close()