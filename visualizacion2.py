import os
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import traceback  # ----- CAMBIO: para imprimir tracebacks

from utils.sample_from_run2 import sample_from_run, parse_config_file
from utils.extract_scores import extract_scores_from_log

def process_run(run_dir, arch_config_path, n_samples=50, mols_per_row=8):
    print(f"\nProcesando {run_dir}...")

    # 1. Cargar config.txt y arch_config.txt
    run_config = {}
    arch_config = {}
    run_config_path = os.path.join(run_dir, "config.txt")
    if os.path.exists(run_config_path):
        run_config = parse_config_file(run_config_path)
    else:
        print(f"  ADVERTENCIA: no existe {run_config_path} — se usará vacío")

    if os.path.exists(arch_config_path):
        arch_config = parse_config_file(arch_config_path)
    else:
        print(f"  ADVERTENCIA: no existe {arch_config_path} — se usará vacío")

    model_config = {**run_config, **arch_config}  # arch_config sobreescribe run_config

    # 2. Crear carpeta de gráficos
    graph_dir = os.path.join(run_dir, "graficos")
    os.makedirs(graph_dir, exist_ok=True)

    # 3. Extraer scores del log.txt y guardar CSV
    log_path = os.path.join(run_dir, "log.txt")
    csv_path = os.path.join(graph_dir, "scores.csv")
    if os.path.exists(log_path):
        try:
            extract_scores_from_log(log_path, out_csv=csv_path)
        except Exception:
            print(f"  ERROR: al extraer scores de {log_path}")
            traceback.print_exc()  # ----- CAMBIO: mostrar detalle
    else:
        print(f"  ADVERTENCIA: no existe {log_path} — se omitirá extracción de scores")

    # 4. Graficar cada score (si existe CSV)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'epoch' in df.columns:
                epochs = df['epoch']
            else:
                epochs = pd.RangeIndex(start=0, stop=len(df), step=1)

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
                plt.savefig(os.path.join(graph_dir, f"{col}_vs_epoch.png"))
                plt.close()
        except Exception:
            print(f"  ERROR: al graficar {csv_path}")
            traceback.print_exc()  # ----- CAMBIO: mostrar detalle
    else:
        print(f"  ADVERTENCIA: {csv_path} no encontrado — se omiten graficas")

    # 5. Generar moléculas y grid
    png_path = os.path.join(graph_dir, "sampled.png")
    mols = []
    smiles = []
    meta = {}
    try:
        # ----- CAMBIO: envolver en try/except para que fallos aquí no detengan todo el script
        mols, smiles, meta = sample_from_run(
            run_dir,
            model_config=model_config,
            n_samples=n_samples,
            mols_per_row=mols_per_row,
            out_png=png_path
        )
        num_valid = len([m for m in mols if m is not None])
        print(f"Generadas {num_valid} moléculas válidas. Grid guardado en {png_path}")
    except Exception:
        print(f"  ERROR: al generar moléculas desde {run_dir} (sample_from_run). Se omitirá esta parte y se continuará con el siguiente run.")
        traceback.print_exc()  # ----- CAMBIO: mostrar detalle
        mols = []  # asegurar que mols es iterable más abajo

    print(f"CSV de scores guardado en {csv_path if os.path.exists(csv_path) else 'NO_EXISTE'}")
    print(f"Gráficas guardadas en {graph_dir}/")

    # 6. Guardar moléculas generadas en SDF
    sdf_path = os.path.join(graph_dir, "sampled.sdf")
    if mols:
        try:
            writer = Chem.SDWriter(sdf_path)
            for m in mols:
                if m is not None:
                    writer.write(m)
            writer.close()
            print(f"SDF guardado en {sdf_path}")
        except Exception:
            print(f"  ERROR: al guardar SDF en {sdf_path}")
            traceback.print_exc()  # ----- CAMBIO: mostrar detalle
            # intentar cerrar writer si existe (seguro)
            try:
                writer.close()
            except Exception:
                pass
    else:
        print(f"No hay moléculas válidas para guardar en SDF para {run_dir}.")

def main():
    # ---------------- CONFIGURACIÓN PRINCIPAL ----------------
    ARCH_NUM = 3  # <--- Cambia aquí el número de arquitectura
    N_SAMPLES = 50
    MOL_PER_ROW = 8

    arch_dir = os.path.join("Model_Output", "Arquitectura", f"Arch_{ARCH_NUM}")
    arch_config_path = os.path.join(arch_dir, "arch_config.txt")

    if not os.path.exists(arch_dir):
        print(f"ERROR: No existe directorio de arquitectura: {arch_dir}")
        return

    # Iterar por todas las carpetas Run_# dentro del Arch_#
    for run_name in sorted(os.listdir(arch_dir)):
        run_path = os.path.join(arch_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        if not run_name.startswith("Run_"):
            continue
        try:
            process_run(run_path, arch_config_path, n_samples=N_SAMPLES, mols_per_row=MOL_PER_ROW)
        except Exception:
            # ----- CAMBIO: protección adicional en caso de error no capturado dentro de process_run
            print(f"ERROR FATAL al procesar {run_path} — se salta y continúa con el siguiente run.")
            traceback.print_exc()

if __name__ == "__main__":
    main()
