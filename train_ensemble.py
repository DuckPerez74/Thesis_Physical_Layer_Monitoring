import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- CONFIGURAÇÕES ---
INPUT_FILE = "train_2h.csv" # Garante que este ficheiro existe na pasta!
EPOCHS = 30 # Reduzido para evitar overfitting
BATCH_SIZE = 64
LEARNING_RATE = 0.001

print(f">>> Treino Robusto (Anti-Overfitting) iniciado...")
df = pd.read_csv(INPUT_FILE, header=None, on_bad_lines='skip')



scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df.values)
joblib.dump(scaler, "scaler_pytorch.gz") # Isto substitui o antigo!

data_scaled = scaler.transform(df.values).astype(np.float32) # ADICIONA ESTA LINHA!


X_train, X_val = train_test_split(data_scaled, test_size=0.1)

train_tensor = torch.tensor(X_train)
val_tensor = torch.tensor(X_val)
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=BATCH_SIZE, shuffle=True)

class RobustAE(nn.Module):
    def __init__(self, variant, input_dim):
        super(RobustAE, self).__init__()
        # Aumentamos o Dropout para 0.3 para forçar a generalização
        drop = 0.3
        
        if variant == 1:
            self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 32), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))
        elif variant == 2:
            self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))
        else: # Variantes simplificadas para não decorar
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, 16), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))

    def forward(self, x):
        # ADICIONAMOS RUÍDO GAUSSIANO NO INPUT (Denoising)
        # Isto impede o overfitting: a rede aprende a limpar o ruído
        if self.training:
            noise = torch.randn_like(x) * 0.05 
            x = x + noise
        return self.decoder(self.encoder(x))

def train_model(idx, var, dim):
    model = RobustAE(var, dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), f"ae_model_{idx}.pth")
    print(f"Modelo {idx} treinado.")

for i in range(1, 6):
    train_model(i, (i%3)+1, data_scaled.shape[1])