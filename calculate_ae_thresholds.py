import torch
import pandas as pd
import numpy as np
import joblib
import torch.nn as nn
import os

# --- 1. DEFINIÇÃO DA ARQUITETURA (Igual ao Treino) ---
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

# --- 2. VERIFICAÇÃO DE FICHEIROS ---
INPUT_FILE = "features_normal_validacao.csv" # Garante que este ficheiro existe na pasta!

if not os.path.exists(INPUT_FILE):
    print(f"ERRO: Não encontrei o ficheiro {INPUT_FILE}!")
    exit()

print(">>> A carregar Scaler e dados de treino...")
scaler = joblib.load("scaler_pytorch.gz")
df = pd.read_csv(INPUT_FILE, header=None, on_bad_lines='skip')
data_scaled = scaler.transform(df.values).astype(np.float32)
input_dim = data_scaled.shape[1]

# --- 3. CÁLCULO DOS LIMITES ---
thresholds = []

for i in range(1, 6):
    variant = (i % 3) + 1
    model_path = f"ae_model_{i}.pth"
    
    if not os.path.exists(model_path):
        print(f"ERRO: O modelo {model_path} não foi encontrado!")
        continue

    model = RobustAE(variant, input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(data_scaled)
        outputs = model(inputs)
        mse = torch.mean(torch.pow(inputs - outputs, 2), dim=1).numpy()
    
    # Em vez de np.percentile(mse, 99.9) que é muito rígido:
    #thr = np.mean(mse) + (3.0 * np.std(mse))
    thr = np.max(mse) * 1.5 
    thresholds.append(thr)
    print(f"Modelo {i} [Variante {variant}] -> Threshold calculado: {thr:.6f}")

# --- 4. GRAVAR FICHEIRO FINAL ---
if len(thresholds) == 5:
    np.save("ensemble_thresholds.npy", np.array(thresholds))
    print("\n[SUCESSO] Ficheiro 'ensemble_thresholds.npy' gerado e guardado!")
else:
    print("\n[ERRO] Não foram calculados os 5 modelos. Ficheiro não guardado.")