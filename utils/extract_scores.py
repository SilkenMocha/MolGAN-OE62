import re
import ast
import csv
import sys
import os

def extract_scores_from_log(log_path, out_csv=None):
    epoch_re = re.compile(r'Epochs\s+(\d+)/(\d+)')
    validation_start_re = re.compile(r'Validation\s*-->\s*{')
    validation_line_re = re.compile(r'Validation\s*-->\s*({.*})')
    rows = []
    current_epoch = None

    with open(log_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Buscar la época
        m_epoch = epoch_re.search(line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))

        # Buscar línea de validación (dict en una línea)
        m_val_line = validation_line_re.search(line)
        if m_val_line and current_epoch is not None:
            dict_str = m_val_line.group(1)
            # Reemplaza nan por None antes de evaluar
            dict_str = re.sub(r'\bnan\b', 'None', dict_str)
            try:
                scores = ast.literal_eval(dict_str)
            except Exception as e:
                print(f"Error al parsear dict en epoch {current_epoch}: {e}")
                i += 1
                continue
            scores['epoch'] = current_epoch
            rows.append(scores)
            current_epoch = None
            i += 1
            continue

        # Bloque multi-línea
        if validation_start_re.search(line) and current_epoch is not None:
            dict_lines = ['{']
            i += 1
            # Leer hasta la línea que contiene solo '}'
            while i < len(lines):
                lnext = lines[i].rstrip()
                dict_lines.append(lnext)
                if lnext.strip().endswith('}'):
                    break
                i += 1
            dict_str = '\n'.join(dict_lines)
            # Reemplaza nan por None antes de evaluar
            dict_str = re.sub(r'\bnan\b', 'None', dict_str)
            try:
                scores = ast.literal_eval(dict_str)
            except Exception as e:
                print(f"Error al parsear dict multi-línea en epoch {current_epoch}: {e}")
                i += 1
                continue
            scores['epoch'] = current_epoch
            rows.append(scores)
            current_epoch = None
        i += 1

    # Ordenar por época
    rows = sorted(rows, key=lambda r: r['epoch'])
    # Determinar todas las claves posibles
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    header = ['epoch'] + sorted([k for k in all_keys if k != 'epoch'])

    if out_csv is None:
        out_csv = os.path.splitext(log_path)[0] + '_scores.csv'
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        writer.writeheader()
        for r in rows:
            row_fixed = {k: ("" if (v is None or str(v) == "nan") else v) for k, v in r.items()}
            writer.writerow(row_fixed)
    print(f"CSV guardado en {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python utils/extract_log_scores.py <ruta_a_log.txt>")
        sys.exit(1)
    log_path = sys.argv[1]
    extract_scores_from_log(log_path)
