#!/bin/bash

echo "Iniciando entrenamiento MolGAN-GAP"
bash /home/erick/MolGAN/run_sh.sh
sleep 30

echo "Iniciando entrenamiento Pre/entrenamiento MolGAN-OE62 sdf"
bash /home/erick/MolGAN-OE62/run_sh1.sh
sleep 30

echo "Iniciando entrenamiento Pre/entrenamiento MolGAN-OE62 smiles"
bash /home/erick/MolGAN-OE62/run_sh2.sh
