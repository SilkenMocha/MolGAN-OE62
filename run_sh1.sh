#!/bin/bash

# Carpeta de salida para preentrenamiento y entrenamiento final
PRETRAIN_OUTDIR="Model_Output/Pretrained_Output"
FINAL_OUTDIR="Model_Output/Trained_Output"
mkdir -p "$PRETRAIN_OUTDIR"
mkdir -p "$FINAL_OUTDIR"

# Listas de parámetros
batch_dims=(128)
las=(1)
z_dims=(32 64)
epochs_pretrain=(500)  # epochs para preentrenamiento
epochs_final=(100)     # epochs para entrenamiento final

# Parámetros fijos
dropout=0
n_critic=5
metric="validity,sas"
n_samples=5000

run_id=100
for bd in "${batch_dims[@]}"; do
  for la in "${las[@]}"; do
    for zd in "${z_dims[@]}"; do

      # ---------- PREENTRENAMIENTO ----------
      for ep in "${epochs_pretrain[@]}"; do
        echo "Preentrenamiento $run_id empezado: bd=$bd, la=$la, zd=$zd, epochs=$ep"
        run_dir="$PRETRAIN_OUTDIR/Run_$run_id"
        mkdir -p "$run_dir"
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
          echo "dataset  = data/db_100nodes.sparsedataset"
        } > "$cfg_file"
        log_file="$run_dir/log.txt"
        python3 run_sh1.py \
          --batch_dim "$bd" \
          --la        "$la" \
          --z_dim     "$zd" \
          --dropout   "$dropout" \
          --n_critic  "$n_critic" \
          --metric    "$metric" \
          --n_samples "$n_samples" \
          --epochs    "$ep" \
          --directory "$run_dir" \
          --dataset   "data/db_100nodes.sparsedataset" \
          > "$log_file" 2>&1
        echo "Preentrenamiento $run_id completado."
      done

      # ---------- ENTRENAMIENTO FINAL ----------
      for ep in "${epochs_final[@]}"; do
        echo "Entrenamiento final $run_id empezado: bd=$bd, la=$la, zd=$zd, epochs=$ep"
        run_dir="$FINAL_OUTDIR/Run_$run_id"
        mkdir -p "$run_dir"
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
          echo "dataset  = data/oe62_sdfnodes.sparsedataset"
          echo "pretrained_dir = $PRETRAIN_OUTDIR/Run_$run_id"
        } > "$cfg_file"
        log_file="$run_dir/log.txt"
        python3 run_sh1.py \
          --batch_dim "$bd" \
          --la        "$la" \
          --z_dim     "$zd" \
          --dropout   "$dropout" \
          --n_critic  "$n_critic" \
          --metric    "$metric" \
          --n_samples "$n_samples" \
          --epochs    "$ep" \
          --directory "$run_dir" \
          --dataset   "data/oe62_sdfnodes.sparsedataset" \
          --pretrained_dir "$PRETRAIN_OUTDIR/Run_$run_id" \
          > "$log_file" 2>&1
        echo "Entrenamiento final $run_id completado."
      done

      run_id=$((run_id + 1))

    done
  done
done

echo "Todas las corridas han finalizado. Carpetas en '$PRETRAIN_OUTDIR/' y '$FINAL_OUTDIR/'."
