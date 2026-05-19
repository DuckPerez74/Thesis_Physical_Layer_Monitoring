import pandas as pd

input_file = "features_baseline_3h.csv"
df = pd.read_csv(input_file, header=None, on_bad_lines='skip')

# Calcular o ponto de corte (2/3 para treino, 1/3 para teste)
cut_point = int(len(df) * 0.66)

train_df = df.iloc[:cut_point]
test_df = df.iloc[cut_point:]

train_df.to_csv("train_2h.csv", index=False, header=None)
test_df.to_csv("test_1h.csv", index=False, header=None)

print(f"Sucesso! Treino: {len(train_df)} linhas, Teste: {len(test_df)} linhas.")