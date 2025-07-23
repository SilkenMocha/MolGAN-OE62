#!/bin/bash
set -euo pipefail
source ~/anaconda3/bin/activate molgan

# Carpeta de salida
OUTDIR="Model_Output"
mkdir -p "$OUTDIR"

# Listas de parámetros
batch_dims=(128 64)
las=1
#las=(0 0.125 0.25 0.375 0.5 0.625 0.75 0.825 1)
z_dims=8
#z_dims=(8 16 32 64)
# Parámetros fijos
dropout=0
n_critic=5
metric="validity,sas"
n_samples=1000
epochs=100

run_id=1
for bd in "${batch_dims[@]}"; do
  for la in "${las[@]}"; do
    for zd in "${z_dims[@]}"; do

      # 1) Crear carpeta de la corrida
      run_dir="$OUTDIR/Run_$run_id"
      mkdir -p "$run_dir"

      # 2) Generar archivo de configuración dentro de Run_<n>
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
        echo "epochs   = $epochs"
      } > "$cfg_file"

      # 3) Ejecutar entrenamiento y guardar log dentro de la misma carpeta
      log_file="$run_dir/log.txt"
      python3 example2.py \
        --batch_dim "$bd" \
        --la        "$la" \
        --z_dim     "$zd" \
        --dropout   "$dropout" \
        --n_critic  "$n_critic" \
        --metric    "$metric" \
        --n_samples "$n_samples" \
        --epochs    "$epochs" \
        > "$log_file" 2>&1

      echo "✅ Corrida $run_id completada: bd=$bd, la=$la, zd=$zd (archivos en $run_dir)"
      run_id=$((run_id + 1))

    done
  done
done

echo "Todas las corridas han finalizado. Carpetas en '$OUTDIR/'."
