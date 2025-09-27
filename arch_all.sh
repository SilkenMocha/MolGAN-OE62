#!/usr/bin/env bash
# run_archs_simple.sh
# Uso: bash run_archs_simple.sh 2 3 4 5 6
set -euo pipefail

SCRIPT="arch_1.sh"

if [ ! -f "$SCRIPT" ]; then
  echo "No se encontró $SCRIPT en $(pwd). Coloca este wrapper junto a Arch_1.sh." >&2
  exit 1
fi

# Ejecutar Arch_1.sh primero
echo "=== Ejecutando Arch_1.sh ==="
bash "$SCRIPT"
echo "=== Arch_1.sh completado ==="

# Si no se pasaron argumentos, salir
if [ $# -eq 0 ]; then
  echo "No se pasaron arquitecturas adicionales. Fin."
  exit 0
fi

# Loop sobre argumentos (números de Arch a ejecutar)
for arch_num in "$@"; do
  # saltar 1 porque ya fue ejecutado
  if [ "$arch_num" = "1" ]; then
    echo "[INFO] Arch_1 ya ejecutado. Saltando 1."
    continue
  fi

  # crear copia temporal con solo el número cambiado en ARCHDIR="...Arch_<n>"
  tmp="$(mktemp /tmp/arch_copy.XXXXXX.sh)"
  sed -E 's/(ARCHDIR="[^"]*Arch_)[0-9]+(")/\1'"${arch_num}"'\2/' "$SCRIPT" > "$tmp"
  chmod +x "$tmp"

  echo "=== Ejecutando Arch_${arch_num} (copia temporal) ==="
  bash "$tmp"
  rc=$?
  rm -f "$tmp"

  if [ $rc -ne 0 ]; then
    echo "Ejecución para Arch_${arch_num} terminó con código $rc. Aborting." >&2
    exit $rc
  fi

  echo "=== Arch_${arch_num} completado ==="
done

echo "=== Todas las arquitecturas solicitadas han finalizado ==="
exit 0
