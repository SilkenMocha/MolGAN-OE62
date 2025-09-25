#!/bin/bash

# Carpeta de arquitectura
ARCHDIR="Model_Output/Arquitectura/Arch_1"
mkdir -p "$ARCHDIR"
cp "$ARCHDIR/arch_config.txt" "$ARCHDIR"

# Listas de parámetros (modifica según tus necesidades)
epochs_list=(20)
dropout_list=(0.0)
learning_rate_list=(1e-4)
batch_dim_list=(128)
z_dim_list=(8 16 32)
n_samples_list=(5000)
n_critic_list=(2)
la_list=(1)

run_id=1
for epochs in "${epochs_list[@]}"; do
  for dropout in "${dropout_list[@]}"; do
    for learning_rate in "${learning_rate_list[@]}"; do
      for batch_dim in "${batch_dim_list[@]}"; do
        for z_dim in "${z_dim_list[@]}"; do
          for n_samples in "${n_samples_list[@]}"; do
            for n_critic in "${n_critic_list[@]}"; do
              for la in "${la_list[@]}"; do
                echo "Corrida $run_id empezada: epochs=$epochs, dropout=$dropout, lr=$learning_rate, bd=$batch_dim, zd=$z_dim, ns=$n_samples, nc=$n_critic, la=$la"
                run_dir="$ARCHDIR/Run_$run_id"
                mkdir -p "$run_dir"

                cfg_file="$run_dir/config.txt"
                {
                  echo "run_id   = $run_id"
                  echo "epochs   = $epochs"
                  echo "dropout  = $dropout"
                  echo "learning_rate = $learning_rate"
                  echo "batch_dim= $batch_dim"
                  echo "z_dim    = $z_dim"
                  echo "n_samples= $n_samples"
                  echo "n_critic = $n_critic"
                  echo "la       = $la"
                } > "$cfg_file"

                log_file="$run_dir/log.txt"
                python3 arch.py \
                  --epochs "$epochs" \
                  --dropout "$dropout" \
                  --learning_rate "$learning_rate" \
                  --batch_dim "$batch_dim" \
                  --z_dim "$z_dim" \
                  --n_samples "$n_samples" \
                  --n_critic "$n_critic" \
                  --la "$la" \
                  --directory "$run_dir" \
                  --arch_config "$ARCHDIR/arch_config.txt" \
                  > "$log_file" 2>&1

                echo "Corrida $run_id completada"
                run_id=$((run_id + 1))
              done
            done
          done
        done
      done
    done
  done
done

echo "Todas las corridas han finalizado. Carpetas en '$ARCHDIR/'."
