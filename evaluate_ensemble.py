import torch
import pandas as pd
import numpy as np
import joblib
import torch.nn as nn
import os
from sklearn.metrics import f1_score



# --- 1. DEFINIÇÃO DA ARQUITETURA ROBUSTA (Igual ao Treino) ---
class RobustAE(nn.Module):
    def __init__(self, variant, input_dim):
        super(RobustAE, self).__init__()
        drop = 0.3
        if variant == 1:
            self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 32), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))
        elif variant == 2:
            self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, 16), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 2. CARREGAR RECURSOS ---
print(">>> A carregar Scaler e Thresholds...")
try:
    scaler = joblib.load("scaler_pytorch.gz")
    thresholds = np.load("ensemble_thresholds.npy")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar os ficheiros base. Já correu o script de treino e de thresholds?\n{e}")
    exit()

def evaluate_file(filename):
    if not os.path.exists(filename):
        print(f"Erro: Ficheiro {filename} não encontrado.")
        return

    # Ler dados
    df = pd.read_csv(filename, header=None, on_bad_lines='skip')
    data_raw = df.values.astype(np.float32)
    
    # Normalizar
    data_scaled = scaler.transform(data_raw)

    print(f"[DEBUG] Stats dos dados após Scaler:")
    print(f"Min: {data_scaled.min():.4f} | Max: {data_scaled.max():.4f}")
    print(f"Média: {np.mean(data_scaled):.4f} | Std: {np.std(data_scaled):.4f}")

    inputs = torch.tensor(data_scaled)
    
    num_samples = len(data_raw)
    input_dim = data_raw.shape[1]
    
    # Matriz para votos
    votes = np.zeros((num_samples, 5))

    print(f"\n[*] ANALISANDO: {filename} ({num_samples} amostras)")
    print("-" * 50)

    # Passar pelos 5 modelos
    for i in range(1, 6):
        variant = (i % 3) + 1
        model = RobustAE(variant, input_dim)
        model.load_state_dict(torch.load(f"ae_model_{i}.pth"))
        model.eval()
        
        with torch.no_grad():
            outputs = model(inputs)
            # Calcular o MSE para cada linha
            mse_per_sample = torch.mean(torch.pow(inputs - outputs, 2), dim=1).numpy()
            
            # DEBUG PRINTS: Para sabermos o que cada modelo está a pensar
            avg_mse = np.mean(mse_per_sample)
            thr = thresholds[i-1]

            #DEBUG
            print(f"Modelo {i} (Var {variant}): MSE Médio = {avg_mse:.6f} | Limiar = {thr:.6f}")

            
            # Guardar voto
            votes[:, i-1] = (mse_per_sample > thr).astype(int)
            
    # --- VOTAÇÃO POR MAIORIA ---
    final_votes = np.sum(votes, axis=1)
    anomalies_detected = np.sum(final_votes >= 3)
    detection_rate = (anomalies_detected / num_samples) * 100

    # --- NOVO: CÁLCULO DE F1-SCORE ---
    # Se o nome do ficheiro tiver "attack", assumimos que a classe real é 1 (anomalia)
    y_true = np.ones(num_samples) if "attack" in filename.lower() else np.zeros(num_samples)
    y_pred = (final_votes >= 3).astype(int) 
    score = f1_score(y_true, y_pred)

    print("-" * 50)
    print(f">> RESULTADO FINAL DO ENSEMBLE <<")
    print(f"Anomalias: {anomalies_detected} / {num_samples}")
    print(f"TAXA DE DETEÇÃO: {detection_rate:.2f}%")
    print(f"F1-SCORE: {score:.4f}") # <--- AQUI ESTÁ A MÉTRICA NOVA
    print("-" * 50)

if __name__ == "__main__":
    while True:
        target = input("\nNome do ficheiro CSV (ou 'sair'): ")
        if target.lower() == 'sair':
            break
        evaluate_file(target)