#!/usr/bin/env python3

import os
import argparse
import pickle
import warnings

import numpy as np
import tensorflow as tf

from rdkit import Chem
from rdkit.Chem import Draw

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import samples
from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn
from optimizers.gan import GraphGANOptimizer

# --- Funciones utilitarias ---

def parse_config_file(cfg_path):
    """Lee un archivo tipo config.txt o arch_config.txt y lo convierte a dict."""
    cfg = {}
    if not os.path.exists(cfg_path):
        return cfg
    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = [p.strip() for p in line.split('=', 1)]
                # intenta convertir a int o float cuando posible
                if v.isdigit():
                    v_conv = int(v)
                else:
                    try:
                        v_conv = float(v)
                    except Exception:
                        v_conv = v
                cfg[k] = v_conv
    return cfg

def resolve_decoder(name):
    mapping = {
        'decoder_adj': decoder_adj,
        'decoder_dot': decoder_dot,
        'decoder_rnn': decoder_rnn,
        'adj': decoder_adj,
        'dot': decoder_dot,
        'rnn': decoder_rnn,
    }
    return mapping.get(str(name), decoder_dot)

def resolve_discriminator(name):
    mapping = {
        'encoder_rgcn': encoder_rgcn,
        'rgcn': encoder_rgcn,
        'encoder': encoder_rgcn,
    }
    return mapping.get(str(name), encoder_rgcn)

def guardar_moleculas_grid(mols, mols_por_fila=5, nombre_archivo="output.png", subImgSize=(200, 200)):
    mols = [m for m in mols if m is not None]
    if len(mols) == 0:
        raise RuntimeError("No hay moléculas válidas para guardar.")
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_por_fila,
        subImgSize=subImgSize,
        useSVG=False,
        returnPNG=False
    )
    outdir = os.path.dirname(nombre_archivo)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    img.save(nombre_archivo)

def mols_to_smiles(mols):
    """Convierte lista de RDKit Mol a SMILES (puede contener None)."""
    smiles = []
    for m in mols:
        if m is None:
            smiles.append(None)
        else:
            try:
                smiles.append(Chem.MolToSmiles(m))
            except Exception:
                smiles.append(None)
    return smiles

# --- Función principal funcional y flexible ---

def sample_from_run(
    run_dir,
    model_config=None,           # dict con hiperparámetros de arquitectura; si None, se lee de config files
    n_samples=100,
    mols_per_row=10,
    out_png=None,
    z_dim=None
):
    """
    Carga el modelo de run_dir, reconstruye usando model_config (o config files), genera n_samples moléculas.
    Devuelve mols, smiles, meta.
    """
    tf.reset_default_graph()
    # 1. Dataset (siempre igual, puede hacerse flexible si lo pides)
    data = SparseMolecularDataset()
    data.load(model_config.get('data.load', 'data/oe62_35sdfnodes.sparsedataset') if model_config else 'data/oe62_35sdfnodes.sparsedataset')

    # 2. Configuración del modelo
    if model_config is None:
        # Fallback: leer config.txt y arch_config.txt si existen
        cfg = parse_config_file(os.path.join(run_dir, 'config.txt'))
        arch_cfg_path = os.path.join(os.path.dirname(run_dir), 'arch_config.txt')
        arch_cfg = parse_config_file(arch_cfg_path)
        # Unir configs, arch_config sobreescribe config.txt
        model_config = {**cfg, **arch_cfg}

    # Prioridad de parámetros: argumento directo > config file > default
    z_dim_final = z_dim if z_dim is not None else int(model_config.get('z_dim', 8))
    decoder_fn = resolve_decoder(model_config.get('decoder', 'decoder_dot'))
    discriminator_fn = resolve_discriminator(model_config.get('discriminator', 'encoder_rgcn'))
    decoder_units = eval(str(model_config.get('decoder_units', '(256, 512, 512)')))
    discriminator_units = eval(str(model_config.get('discriminator_units', '((256, 128), 256, (256, 128))')))
    soft_gumbel_softmax = str(model_config.get('soft_gumbel', False)) == "True"
    hard_gumbel_softmax = str(model_config.get('hard_gumbel', False)) == "True"
    batch_discriminator = str(model_config.get('batch_discriminator', True)) == "True"
    feature_matching = str(model_config.get('feature_matching', True)) == "True"
    learning_rate = float(model_config.get('learning_rate', 1e-4))

    # 3. Reconstrucción del modelo
    model = GraphGANModel(
        data.vertexes,
        data.bond_num_types,
        data.atom_num_types,
        z_dim_final,
        decoder_units=decoder_units,
        discriminator_units=discriminator_units,
        decoder=decoder_fn,
        discriminator=discriminator_fn,
        soft_gumbel_softmax=soft_gumbel_softmax,
        hard_gumbel_softmax=hard_gumbel_softmax,
        batch_discriminator=batch_discriminator
    )
    optimizer = GraphGANOptimizer(model, learning_rate=learning_rate, feature_matching=feature_matching)

    # 4. Restaurar sesión y checkpoint
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No se encontró checkpoint en {run_dir}.")
    print(f"Restaurando checkpoint {ckpt} ...")
    saver.restore(session, ckpt)
    print("Checkpoint restaurado OK.")

    # 5. Generar moléculas
    z = model.sample_z(n_samples)
    mols = samples(data, model, session, z, sample=True)

    # 6. Guardar imagen si se pide
    if out_png:
        guardar_moleculas_grid(mols, mols_por_fila=mols_per_row, nombre_archivo=out_png)

    # 7. Obtener smiles
    smiles = mols_to_smiles(mols)

    # 8. Devolver metadatos usados
    meta = {
        "z_dim": z_dim_final,
        "decoder_units": decoder_units,
        "discriminator_units": discriminator_units,
        "decoder": decoder_fn.__name__,
        "discriminator": discriminator_fn.__name__,
        "soft_gumbel_softmax": soft_gumbel_softmax,
        "hard_gumbel_softmax": hard_gumbel_softmax,
        "batch_discriminator": batch_discriminator,
        "feature_matching": feature_matching,
        "learning_rate": learning_rate,
        "run_dir": run_dir
    }
    return mols, smiles, meta

# --- CLI para casos legacy (puedes quitarlo si solo usas como módulo) ---

def main():
    parser = argparse.ArgumentParser(description="Sample molecules from a trained run directory.")
    parser.add_argument("--run_dir", required=True, help="Carpeta que contiene checkpoint/model.ckpt.* y config.txt")
    parser.add_argument("--n_samples", type=int, default=100, help="Número de moléculas a muestrear")
    parser.add_argument("--out_png", type=str, default=None, help="Ruta para guardar la imagen de salida (png). Si no, no guarda imagen.")
    parser.add_argument("--mols_per_row", type=int, default=10, help="Moléculas por fila en la imagen")
    parser.add_argument("--print_smiles", action="store_true", help="Imprimir SMILES por consola")
    parser.add_argument("--z_dim", type=int, default=None, help="Forzar z_dim si no hay config/trainer.pkl o quieres sobreescribirla")
    args = parser.parse_args()

    mols, smiles, meta = sample_from_run(
        args.run_dir,
        n_samples=args.n_samples,
        mols_per_row=args.mols_per_row,
        out_png=args.out_png,
        z_dim=args.z_dim
    )

    if args.print_smiles:
        for i, s in enumerate(smiles):
            print(f"{i+1}\t{s}")

    if args.out_png:
        print(f"Imagen guardada en: {args.out_png}")
    print(f"Se generaron {len([m for m in mols if m is not None])} moléculas válidas de {len(mols)}.")
    print("Metadatos usados para reconstrucción:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
