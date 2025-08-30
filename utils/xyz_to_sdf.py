#!/usr/bin/env python3
"""
MolGAN-OE62/utils/xyz_to_sdf.py

Lee data/m1507656/df_62k.json (pandas orient='split'), extrae xyz_pbe_relaxed,
genera data/m1507656/all_molecules.xyz y (si obabel está disponible) data/m1507656/all_molecules.sdf

Uso:
    python3 utils/xyz_to_sdf.py
Opciones:
    --input PATH        (default: data/m1507656/df_62k.json)
    --out-xyz PATH      (default: data/m1507656/all_molecules.xyz)
    --out-sdf PATH      (default: data/m1507656/all_molecules.sdf)
    --max N             (solo procesar N moléculas, por debugging)
"""
import os
import re
import argparse
import json
import pandas as pd
import subprocess
from pathlib import Path

def parse_xyz_block(xyz_text):
    """
    Extrae número de átomos y las líneas atómicas de un bloque que viene como string.
    Retorna (natoms_declared, atom_lines_list).
    Se intenta ser tolerante si el contenido está ligeramente desordenado.
    """
    if xyz_text is None:
        return None, []
    # Asegurar str
    s = str(xyz_text)

    # Buscar un entero al principio (número de átomos)
    m = re.search(r'^\s*(\d+)\s*[\r\n]+(.*)$', s, re.DOTALL)
    if m:
        try:
            natoms = int(m.group(1))
        except ValueError:
            natoms = None
        rest = m.group(2)
    else:
        # Si no encuentra encabezado, intentar tomar todas las líneas no vacías
        natoms = None
        rest = s

    # Tomar solo líneas no vacías
    lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]

    # Normalizar cada línea: tomar los primeros 4 tokens (elemento x y z)
    atom_lines = []
    for ln in lines:
        toks = re.split(r'\s+', ln)
        if len(toks) >= 4:
            atom_lines.append(' '.join(toks[:4]))
        else:
            # si la línea no tiene 4 tokens, la ignoramos (o la añadimos tal cual para inspección)
            atom_lines.append(' '.join(toks))

    return natoms, atom_lines

def main(args):
    input_path = Path(args.input)
    out_xyz = Path(args.out_xyz)
    out_sdf = Path(args.out_sdf)

    if not input_path.exists():
        print(f"Error: no existe el archivo de entrada {input_path.resolve()}")
        return

    # Cargar JSON con pandas (orient='split')
    print("Leyendo JSON (pandas orient='split') ...")
    df = pd.read_json(input_path, orient='split')

    # Verificar columna
    if 'xyz_pbe_relaxed' not in df.columns:
        print("Error: la columna 'xyz_pbe_relaxed' no está en el dataframe.")
        print("Columnas disponibles:", df.columns.tolist())
        return

    out_xyz.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_xyz.open('w', encoding='utf-8') as f:
        for i, row in df.iterrows():
            if args.max and written >= args.max:
                break

            refcode = row.get('refcode_csd', f'row_{i}')
            xyz_text = row.get('xyz_pbe_relaxed', None)
            if not xyz_text or (isinstance(xyz_text, float) and pd.isna(xyz_text)):
                skipped += 1
                continue

            natoms_declared, atom_lines = parse_xyz_block(xyz_text)

            if not atom_lines:
                skipped += 1
                continue

            # Si natoms_declared está presente pero no coincide, avisar en stdout y usar la cantidad real disponible.
            natoms_to_write = len(atom_lines)
            if natoms_declared is not None and natoms_declared != natoms_to_write:
                print(f"Advertencia: fila {i} refcode={refcode} natoms_declared={natoms_declared} pero se encontraron {natoms_to_write} líneas atómicas. Escribiendo {natoms_to_write}.")

            # Escribir bloque XYZ estándar: línea 1 = natoms, línea 2 = comentario (refcode), luego las líneas atómicas
            f.write(f"{natoms_to_write}\n")
            f.write(f"{refcode}\n")
            for ln in atom_lines[:natoms_to_write]:
                f.write(ln + "\n")

            written += 1

    print(f"Escrito {written} moléculas en {out_xyz.resolve()}. Saltadas: {skipped}.")

    # Intentar convertir con obabel
    # Comando simple: obabel input.xyz -O output.sdf
    obabel_cmd = None
    for cmd in ('obabel', 'babel'):
        try:
            subprocess.run([cmd, '-V'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            obabel_cmd = cmd
            break
        except FileNotFoundError:
            continue

    if obabel_cmd is None:
        print("\nNo se encontró 'obabel' ni 'babel' en PATH.")
        print("Instala OpenBabel (ej. con conda: conda install -c conda-forge openbabel) o ejecuta manualmente:")
        print(f"    obabel {out_xyz} -O {out_sdf}")
        return

    print(f"\nConvirtiendo a SDF usando '{obabel_cmd}' ... (esto puede tardar según el número de moléculas)")
    # Ejecutar conversión; no añadir flags extra para dejar que OpenBabel detecte conectividades a partir de las coordenadas.
    try:
        subprocess.run([obabel_cmd, str(out_xyz), '-O', str(out_sdf)], check=True)
        print(f"Conversion completada: {out_sdf.resolve()}")
    except subprocess.CalledProcessError as e:
        print("obabel devolvió un error:", e)
        print("Puedes intentar ejecutar manualmente:")
        print(f"    {obabel_cmd} {out_xyz} -O {out_sdf}")
    except FileNotFoundError:
        print("Error inesperado: obabel no encontrado al intentar ejecutarlo.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convertir xyz_pbe_relaxed (JSON orient='split') -> XYZ multi-molécula -> SDF con OpenBabel")
    parser.add_argument('--input', default='data/m1507656/df_62k.json', help='Ruta al JSON (orient=split)')
    parser.add_argument('--out-xyz', default='data/OE62.xyz', help='Archivo XYZ de salida')
    parser.add_argument('--out-sdf', default='data/OE62.sdf', help='Archivo SDF de salida (intentado vía obabel)')
    parser.add_argument('--max', type=int, default=0, help='Procesar solo N moléculas (0 = todas)')
    args = parser.parse_args()

    # Normalizar: si max==0 -> None
    if args.max == 0:
        args.max = None

    main(args)
