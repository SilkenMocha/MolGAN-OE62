import os
import matplotlib.pyplot as plt
from utils.sample_from_run import sample_from_run
from utils.extract_scores import extract_scores_from_log
import pandas as pd

#__________________ Configuración global __________________
RUN_ID = "Run_2"  # <--- Cambia esto para el run que quieras
ROOT_DIR = os.path.join("Model_Output", RUN_ID)
GRAPH_DIR = os.path.join(ROOT_DIR, "graficos")
os.makedirs(GRAPH_DIR, exist_ok=True)

#_________________ Generación archivo CSV _________________
LOG_PATH = os.path.join(ROOT_DIR, "log.txt")
CSV_PATH = os.path.join(GRAPH_DIR, "scores.csv")
extract_scores_from_log(LOG_PATH, out_csv=CSV_PATH)

#_________________ Generación de gráficas _________________
df = pd.read_csv(CSV_PATH)
epochs = df['epoch']

for col in df.columns:
    if col == 'epoch':
        continue
    y = pd.to_numeric(df[col], errors='coerce')
    if y.isnull().all():
        continue
    plt.figure()
    plt.plot(epochs, y, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(col)
    plt.title(f"{col} vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{col}_vs_epoch.png"))
    plt.close()

#___________ Visualización de moléculas generadas __________
PNG_PATH = os.path.join(GRAPH_DIR, "sampled.png")
mols, smiles, meta = sample_from_run(ROOT_DIR, n_samples=50, mols_per_row=8, out_png=PNG_PATH)
num_valid = len([m for m in mols if m is not None])
print(f"Generadas {num_valid} moléculas válidas. Grid guardado en {PNG_PATH}")
print(f"CSV de scores guardado en {CSV_PATH}")
print(f"Gráficas guardadas en {GRAPH_DIR}/")
