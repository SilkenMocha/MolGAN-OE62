#!/usr/bin/env python3
"""
scores.py - Combina scores.csv y genera workbook por métrica + hoja 'config' con arch_config.txt y config.txt por run

Características:
 - Opción --no-csv para evitar generar el CSV combinado.
 - Hoja 'config' con filas en el orden solicitado y sin 'la' ni 'run_id'.
 - Columnas: primero Arch_X (arch_config) luego Arch_X/Run_Y (run config).

Uso:
  python3 scores.py
  python3 scores.py --no-csv

Requiere: pandas, openpyxl
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

def find_score_files(root='.', pattern='Arch_*/Run_*/graficos/scores.csv'):
    return sorted(Path(root).glob(pattern))

def extract_ids_from_path(path):
    p = Path(path).resolve()
    arch_id, run_id = None, None
    for part in p.parts[::-1]:
        if run_id is None and part.lower().startswith('run'):
            run_id = part
        elif arch_id is None and part.lower().startswith('arch'):
            arch_id = part
        if arch_id and run_id:
            break
    if arch_id is None or run_id is None:
        for part in p.parts:
            if arch_id is None and 'arch' in part.lower():
                arch_id = part
            if run_id is None and 'run' in part.lower():
                run_id = part
    return arch_id or 'arch_unknown', run_id or 'run_unknown'

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] No se pudo leer {path}: {e}", file=sys.stderr)
        return None

def read_key_value_file(path: Path):
    """
    Lee un archivo con formato 'clave = valor' por línea.
    Devuelve dict clave->valor (strings). Omite líneas vacías o que empiecen con '#'.
    Soporta también líneas con separación por espacios: 'run_id   = 1'.
    """
    if not path.exists():
        return {}
    data = {}
    try:
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    data[k.strip()] = v.strip()
                else:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == '=':
                        data[parts[0].strip()] = " ".join(parts[2:]).strip()
                    else:
                        data[line] = ''
    except Exception as e:
        print(f"[WARN] Error leyendo {path}: {e}", file=sys.stderr)
    return data

def read_arch_config(root_path: Path, arch_id: str):
    return read_key_value_file(root_path / arch_id / 'arch_config.txt')

def read_run_config(root_path: Path, arch_id: str, run_id: str):
    return read_key_value_file(root_path / arch_id / run_id / 'config.txt')

def main(args):
    root = Path(args.root)
    files = find_score_files(root=str(root), pattern=args.pattern)
    if not files:
        print(f"No se encontraron scores.csv con patrón '{args.pattern}' desde {root}.", file=sys.stderr)
        sys.exit(1)

    rows = []
    per_run_dfs = {}      # clave "Arch_X/Run_Y" -> df
    archs_found = set()
    runs_found = []       # lista de tuplas (arch_id, run_id) para orden

    for p in files:
        f = Path(p)
        df = safe_read_csv(f)
        if df is None:
            continue
        arch_id, run_id = extract_ids_from_path(f)
        archs_found.add(arch_id)
        run_key = f"{arch_id}/{run_id}"
        if (arch_id, run_id) not in runs_found:
            runs_found.append((arch_id, run_id))

        # Asegurar columna epoch
        if args.epoch_col not in df.columns:
            alt = None
            for cand in ['epoch', 'step', 'epo', 'epoca', 'epochs']:
                if cand in df.columns:
                    alt = cand
                    break
            if alt:
                df = df.rename(columns={alt: args.epoch_col})
                if args.verbose:
                    print(f"[INFO] Renombrando {alt} -> {args.epoch_col} en {f}")
            else:
                df[args.epoch_col] = df.index.astype(int)
                if args.verbose:
                    print(f"[INFO] Creando columna {args.epoch_col} desde índice en {f}")
        try:
            df[args.epoch_col] = pd.to_numeric(df[args.epoch_col], errors='coerce').astype('Int64')
        except Exception:
            pass

        df_out = df.copy()
        df_out['arch_id'] = arch_id
        df_out['run_id'] = run_id
        rows.append(df_out)

        try:
            tmp = df.set_index(args.epoch_col, drop=False)
        except Exception:
            tmp = df
        per_run_dfs[run_key] = tmp

        if args.verbose:
            print(f"[FOUND] {f} -> {run_key} (rows={len(df)})")

    if not rows:
        print("No se pudieron leer datos válidos desde los scores.csv.", file=sys.stderr)
        sys.exit(1)

    # CSV combinado (opcional)
    if not args.no_csv:
        combined = pd.concat(rows, ignore_index=True, sort=False)
        combined_out = root / args.out_csv
        try:
            combined.to_csv(combined_out, index=False)
            print(f"[OK] CSV combinado escrito en: {combined_out}")
        except Exception as e:
            print(f"[ERROR] No se pudo escribir {combined_out}: {e}", file=sys.stderr)

    # Determinar columnas de métricas (tomamos las de combined o de first row)
    excluded = {args.epoch_col, 'arch_id', 'run_id'}
    reference_df = combined if (not args.no_csv and 'combined' in locals()) else rows[0]
    metric_cols = [c for c in reference_df.columns if c not in excluded]

    if not metric_cols:
        print("[WARN] No se encontraron columnas de métricas en los CSV (solo epoch/ids). Nada que pivotear.", file=sys.stderr)
    else:
        # Recolectar epochs
        all_epochs = set()
        for df in per_run_dfs.values():
            try:
                idx_vals = df.index.dropna().unique()
            except Exception:
                idx_vals = df.index.unique()
            all_epochs.update([int(v) for v in idx_vals if pd.notna(v)])
        epoch_index = pd.Index(sorted(all_epochs), name=args.epoch_col)

        # Construir DataFrames por métrica
        metrics_dict = {}
        for metric in metric_cols:
            df_metric = pd.DataFrame(index=epoch_index)
            for key, df in per_run_dfs.items():
                if metric in df.columns:
                    ser = pd.to_numeric(df[metric], errors='coerce')
                    try:
                        ser.index = ser.index.astype('Int64')
                    except Exception:
                        pass
                    ser = ser.groupby(ser.index).first()
                    df_metric[key] = ser.reindex(epoch_index)
                else:
                    df_metric[key] = pd.Series([pd.NA]*len(epoch_index), index=epoch_index)
            metrics_dict[metric] = df_metric

    # ---------- leer configs de Arch y Run ----------
    arch_configs = {}
    for arch_id in sorted(archs_found):
        arch_configs[arch_id] = read_arch_config(root, arch_id)
    run_configs = {}
    for arch_id, run_id in runs_found:
        run_key = f"{arch_id}/{run_id}"
        run_configs[run_key] = read_run_config(root, arch_id, run_id)

    # Construir DataFrame 'config' con el orden solicitado y sin 'la' ni 'run_id'
    desired_order = [
        'epochs',
        'batch_dim',
        'dropout',
        'la',               # lo incluimos aquí para poder eliminarlo luego
        'learning_rate',
        'n_critic',
        'n_samples',
        'z_dim',
        'data.load',
        'decoder',
        'feature_matching',
        'batch_discriminator',
        'soft_gumbel',
        'hard_gumbel',
        'decoder_units',
        'discriminator_units'
    ]
    # columnas: Arch_X primero, luego Arch_X/Run_Y en orden encontrado
    config_columns = []
    config_map = {}
    for arch_id in sorted(arch_configs.keys()):
        config_columns.append(arch_id)
        config_map[arch_id] = arch_configs.get(arch_id, {})
    for arch_id, run_id in runs_found:
        run_key = f"{arch_id}/{run_id}"
        config_columns.append(run_key)
        config_map[run_key] = run_configs.get(run_key, {})

    # colectar todas las claves presentes
    all_keys = set()
    for cfg in config_map.values():
        all_keys.update(cfg.keys())

    # eliminar 'la' y 'run_id' si aparecen
    all_keys.discard('la')
    all_keys.discard('run_id')

    # construir lista final de keys: las que están en desired_order y en all_keys (en ese orden),
    # luego el resto ordenado alfabeticamente
    ordered_keys = [k for k in desired_order if k in all_keys and k != 'la' and k != 'run_id']
    remaining = sorted(list(all_keys - set(ordered_keys)))
    final_keys = ordered_keys + remaining

    # Si no hay keys (muy raro), poner fila vacía
    if not final_keys:
        final_keys = []

    # construir config_df con esos final_keys
    config_df = pd.DataFrame(
        {col: [config_map[col].get(k, pd.NA) for k in final_keys] for col in config_columns},
        index=final_keys
    )

    # Escribir workbook Excel con hojas por métrica + hoja 'config'
    out_xlsx = root / args.out_xlsx
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
            if 'metrics_dict' in locals():
                for metric, dfm in metrics_dict.items():
                    sheet_name = str(metric)[:31]
                    try:
                        dfm.index = dfm.index.astype(int)
                    except Exception:
                        pass
                    dfm.to_excel(writer, sheet_name=sheet_name)
            # hoja config (si existe)
            if config_df is not None and not config_df.empty:
                config_df.to_excel(writer, sheet_name='config')
        print(f"[OK] Workbook con hojas por métrica (y 'config') escrito en: {out_xlsx}")
    except Exception as e:
        print(f"[ERROR] No se pudo escribir {out_xlsx}: {e}", file=sys.stderr)
        if config_df is not None:
            try:
                cfg_csv = root / 'archs_runs_config.csv'
                config_df.to_csv(cfg_csv)
                print(f"[OK] (fallback) config escrita en: {cfg_csv}")
            except Exception as e2:
                print(f"[ERROR] fallback writing config also failed: {e2}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combina scores.csv y genera un Excel por métrica + hoja 'config' (arch+run).")
    parser.add_argument('--root', '-r', default='.', help='Directorio raíz donde buscar (ejecutar desde Arquitectura recommended).')
    parser.add_argument('--pattern', default='Arch_*/Run_*/graficos/scores.csv', help='Patrón glob relativo para localizar los scores.csv')
    parser.add_argument('--out-csv', default='combined_scores_raw.csv', help='Nombre del CSV combinado de salida')
    parser.add_argument('--out-xlsx', default='scores_by_metric.xlsx', help='Nombre del workbook Excel de salida')
    parser.add_argument('--epoch-col', default='epoch', help='Nombre de la columna que contiene la época (se intentará inferir si no existe).')
    parser.add_argument('--no-csv', action='store_true', help='No generar el CSV combinado')
    parser.add_argument('--verbose', action='store_true', help='Modo verboso')
    args = parser.parse_args()
    main(args)
