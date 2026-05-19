import pandas as pd
import numpy as np

files = ["train_2h.csv", "test_1h.csv", "wifi_normal_casa_validacao.csv"]

for f in files:
    try:
        df = pd.read_csv(f, header=None)
        # Vamos olhar para a primeira coluna (que costuma ser a média do Canal 1)
        mean_val = df[0].mean()
        max_val = df[0].max()
        print(f"Ficheiro: {f}")
        print(f"  - Linhas: {len(df)}")
        print(f"  - Média Global (Col 0): {mean_val:.2f} dB")
        print(f"  - Máximo Absoluto (Col 0): {max_val:.2f} dB")
        print("-" * 30)
    except:
        print(f"Erro ao ler {f}")