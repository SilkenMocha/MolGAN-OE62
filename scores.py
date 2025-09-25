#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from pathlib import Path

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
    return arch_id or 'arch_unknown', run_id or 'run_unknown'

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] No se pudo leer {path}: {e}", file=sys.stderr)
        return None

def main(args):
    files = find_score_files(args.root, args.pattern)
    if not files:
        print("No se encontraron scores.csv", file=sys.stderr)
        sys.exit(1)

    rows, per_run_dfs = [], {}
    for f in files:
        df = safe_read_csv(f)
        if df is None:
            continue
        arch_id, run_id = extract_ids_from_path(f)
        if args.epoch_col not in df.columns:
            df[args.epoch_col] = df.index
        df_out = df.copy()
        df_out['arch_id'] = arch_id
        df_out['run_id'] = run_id
        rows.append(df_out)
        per_run_dfs[f"{arch_id}/{run_id}"] = df.set_index(args.epoch_col, drop=False)

    if not rows:
        print("No se pudieron leer datos válidos", file=sys.stderr)
        sys.exit(1)

    # CSV combinado (opcional)
    if not args.no_csv:
        combined = pd.concat(rows, ignore_index=True, sort=False)
        combined_out = Path(args.root) / args.out_csv
        combined.to_csv(combined_out, index=False)
        print(f"[OK] CSV combinado escrito en: {combined_out}")

    # Identificar métricas
    excluded = {args.epoch_col, 'arch_id', 'run_id'}
    metric_cols = [c for c in rows[0].columns if c not in excluded]

    if not metric_cols:
        print("No hay métricas en los CSV", file=sys.stderr)
        sys.exit(0)

    # Construir Excel
    metrics_dict = {}
    all_epochs = set()
    for df in per_run_dfs.values():
        all_epochs.update(df.index.unique())
    epoch_index = pd.Index(sorted(all_epochs), name=args.epoch_col)

    for metric in metric_cols:
        df_metric = pd.DataFrame(index=epoch_index)
        for key, df in per_run_dfs.items():
            if metric in df.columns:
                ser = pd.to_numeric(df[metric], errors='coerce')
                df_metric[key] = ser.reindex(epoch_index)
        metrics_dict[metric] = df_metric

    out_xlsx = Path(args.root) / args.out_xlsx
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for metric, dfm in metrics_dict.items():
            dfm.to_excel(writer, sheet_name=str(metric)[:31])
    print(f"[OK] Excel por métrica escrito en: {out_xlsx}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='Directorio raíz (default=.)')
    p.add_argument('--pattern', default='Arch_*/Run_*/graficos/scores.csv')
    p.add_argument('--out-csv', default='combined_scores_raw.csv')
    p.add_argument('--out-xlsx', default='scores_by_metric.xlsx')
    p.add_argument('--epoch-col', default='epoch')
    p.add_argument('--no-csv', action='store_true', help='No generar el CSV combinado')
    args = p.parse_args()
    main(args)
