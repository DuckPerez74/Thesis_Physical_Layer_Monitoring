import socket

# Configurações do servidor
IP_ATACANTE = "10.0.2.15" # Ouve em todas as interfaces
PORTA = 8080

def start_listener():
    # Usamos UDP porque é mais comum em exfiltração rápida e C2
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP_ATACANTE, PORTA))
    
    print(f"[*] Atacante online! À espera de dados na porta {PORTA}...")
    
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            print(f"[+] Recebidos {len(data)} bytes de {addr}")
    except KeyboardInterrupt:
        print("\n[!] Listener encerrado.")
    finally:
        sock.close()

if __name__ == "__main__":
    start_listener()