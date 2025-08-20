#!/bin/bash

# Carpeta de salida
OUTDIR="Model_Output"
mkdir -p "$OUTDIR"

# Listas de parámetros
batch_dims=(128)
las=(1)
z_dims=(16 32)
epochs=(50 100 200 300 400 500 600)  # ahora es lista de valores

# Parámetros fijos
dropout=0
n_critic=5
metric="validity,sas"
n_samples=5000

run_id=1
for bd in "${batch_dims[@]}"; do
  for la in "${las[@]}"; do
    for zd in "${z_dims[@]}"; do
      for ep in "${epochs[@]}"; do  # recorrer valores de epochs
        echo "Corrida $run_id empezada: bd=$bd, la=$la, zd=$zd, epochs=$ep"

        # 1) Crear carpeta de la corrida
        run_dir="$OUTDIR/Run_$run_id"
        mkdir -p "$run_dir"

        # 2) Generar archivo de configuración
        cfg_file="$run_dir/config.txt"
        {
          echo "run_id   = $run_id"
          echo "batch_dim= $bd"
          echo "la       = $la"
          echo "z_dim    = $zd"
          echo "dropout  = $dropout"
          echo "n_critic = $n_critic"
          echo "metric   = $metric"
          echo "n_samples= $n_samples"
          echo "epochs   = $ep"
        } > "$cfg_file"

        # 3) Ejecutar entrenamiento
        log_file="$run_dir/log.txt"
        python3 run_sh.py \
          --batch_dim "$bd" \
          --la        "$la" \
          --z_dim     "$zd" \
          --dropout   "$dropout" \
          --n_critic  "$n_critic" \
          --metric    "$metric" \
          --n_samples "$n_samples" \
          --epochs    "$ep" \
          --directory "$run_dir" \
          > "$log_file" 2>&1

        echo "Corrida $run_id completada: bd=$bd, la=$la, zd=$zd, epochs=$ep"
        run_id=$((run_id + 1))

      done
    done
  done
done

echo "Todas las corridas han finalizado. Carpetas en '$OUTDIR/'."
