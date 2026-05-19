import pandas as pd
import numpy as np

# Carrega os dois ficheiros
train = pd.read_csv("train_2h.csv", header=None, nrows=1000)
valid = pd.read_csv("wifi_normal_casa_validacao.csv", header=None, nrows=1000)

print(f"Colunas Treino: {train.shape[1]} | Colunas Validação: {valid.shape[1]}")
print("-" * 30)
print("Média das primeiras 5 colunas (Treino):")
print(train.iloc[:, :5].mean())
print("-" * 30)
print("Média das primeiras 5 colunas (Validação):")
print(valid.iloc[:, :5].mean())