#!/usr/bin/env python3

import os
import argparse
import pickle
import inspect
import types
import warnings

import numpy as np
import tensorflow as tf

from rdkit import Chem
from rdkit.Chem import Draw

# Asume la estructura del repo y que estas rutas son resolubles desde la raíz del repo:
from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.trainer import Trainer
from utils.utils import samples  # samples es la función usada para generar moléculas
from models.gan import GraphGANModel
from models import encoder_rgcn, decoder_adj, decoder_dot, decoder_rnn
from optimizers.gan import GraphGANOptimizer


# ------------------ utilidades para leer config/trainer.pkl ------------------ #
def parse_config(cfg_path):
    """Parsea el config.txt creado por run_sh.sh; devuelve dict con claves y valores."""
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
                        # Si es palabra (ej: "decoder_adj") dejar como string
                        v_conv = v
                cfg[k] = v_conv
    return cfg


def try_load_pickle(path):
    """Intentar cargar un pickle de forma segura, devolviendo None si falla."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        warnings.warn(f"Fallo al cargar pickle {path}: {e}")
        return None


def extract_model_metadata_from_trainer(trainer_obj):
    """
    Heurística para extraer metadatos de un objeto trainer (o dict) cargado desde trainer.pkl.
    Buscamos claves/atributos comunes: z_dim, decoder_units, discriminator_units, decoder, discriminator,
    soft_gumbel_softmax, hard_gumbel_softmax, batch_discriminator.
    Devuelve un dict con lo que encuentre.
    """
    meta = {}

    # Si es un dict, examinar directamente
    if isinstance(trainer_obj, dict):
        search_src = trainer_obj
    else:
        # si es un objeto, intentar obtener __dict__ o atributos públicos
        try:
            search_src = trainer_obj.__dict__
        except Exception:
            # fallback: construir dict con atributos inspeccionables
            search_src = {}
            for name in dir(trainer_obj):
                if name.startswith('_'):
                    continue
                try:
                    val = getattr(trainer_obj, name)
                    # solo incluir tipos simples o dict-likes to avoid cargar grandes grafos
                    if isinstance(val, (int, float, str, tuple, list, dict, bool)):
                        search_src[name] = val
                except Exception:
                    continue

    # keys candidatas que podríamos encontrar
    candidate_keys = [
        'z_dim', 'zdim', 'z_dim_', 'z', 'decoder_units', 'decoder_hidden',
        'discriminator_units', 'disc_units', 'decoder', 'discriminator',
        'soft_gumbel_softmax', 'hard_gumbel_softmax', 'batch_discriminator',
        'decoder_name', 'discriminator_name', 'model_args', 'model_kwargs'
    ]

    # Buscar en la capa superior
    for k in candidate_keys:
        if k in search_src:
            meta[k] = search_src[k]

    # Si hay model_args o model_kwargs, investigar dentro
    for mk in ('model_args', 'model_kwargs', 'model_params', 'params'):
        if mk in search_src and isinstance(search_src[mk], dict):
            for k in candidate_keys:
                if k in search_src[mk]:
                    meta[k] = search_src[mk][k]

    # Intentar buscar un objeto 'model' dentro del trainer_obj y extraer su información
    model_obj = None
    if isinstance(trainer_obj, dict):
        model_obj = trainer_obj.get('model', None)
    else:
        # atributos comunes
        for name in ('model', '_model', 'graph_model'):
            if hasattr(trainer_obj, name):
                try:
                    model_obj = getattr(trainer_obj, name)
                    break
                except Exception:
                    continue

    if model_obj is not None:
        # extraer atributos simples del modelo
        for attr in ('z_dim', 'zdim', 'decoder_units', 'discriminator_units',
                     'soft_gumbel_softmax', 'hard_gumbel_softmax', 'batch_discriminator'):
            if hasattr(model_obj, attr):
                try:
                    meta[attr] = getattr(model_obj, attr)
                except Exception:
                    pass
        # extraer nombre del decoder/discriminator si están presentes como referencias
        for attr in ('decoder', 'discriminator'):
            if hasattr(model_obj, attr):
                val = getattr(model_obj, attr)
                # si es función, tomar su nombre
                if isinstance(val, types.FunctionType):
                    meta[attr] = val.__name__
                else:
                    # si es string, usar directamente
                    meta[attr] = str(val)

    return meta


def resolve_decoder_decodername(name):
    """Mapea un nombre simple a la función real del paquete models."""
    if name is None:
        return None
    name = str(name)
    mapping = {
        'decoder_adj': decoder_adj,
        'decoder_dot': decoder_dot,
        'decoder_rnn': decoder_rnn,
        'adj': decoder_adj,
        'dot': decoder_dot,
        'rnn': decoder_rnn
    }
    return mapping.get(name, None)


def resolve_discriminator_name(name):
    """Mapea un nombre simple a la función real del paquete models (discriminador/encoder)."""
    if name is None:
        return None
    name = str(name)
    mapping = {
        'encoder_rgcn': encoder_rgcn,
        'rgcn': encoder_rgcn,
        'encoder': encoder_rgcn
    }
    return mapping.get(name, None)


# ------------------ funciones principales para reconstruir y samplear ------------------ #
def load_trainer_metadata(run_dir):
    """
    Intenta cargar run_dir/trainer.pkl y devolver metadatos útiles (heurística).
    Devuelve dict (vacío si nada útil).
    """
    trainer_pkl = os.path.join(run_dir, 'trainer.pkl')
    obj = try_load_pickle(trainer_pkl)
    if obj is None:
        return {}
    meta = extract_model_metadata_from_trainer(obj)
    # Normalizar claves:
    normalized = {}
    # normalizar z_dim
    for k in ('z_dim', 'zdim', 'z'):
        if k in meta:
            normalized['z_dim'] = int(meta[k])
            break
    # decoder units
    for k in ('decoder_units',):
        if k in meta:
            normalized['decoder_units'] = tuple(meta[k]) if isinstance(meta[k], (list, tuple)) else meta[k]
            break
    # discriminator units
    for k in ('discriminator_units', 'disc_units'):
        if k in meta:
            normalized['discriminator_units'] = tuple(meta[k]) if isinstance(meta[k], (list, tuple)) else meta[k]
            break
    # decoder / discriminator names
    for k in ('decoder', 'decoder_name'):
        if k in meta:
            normalized['decoder_name'] = str(meta[k])
            break
    for k in ('discriminator', 'discriminator_name'):
        if k in meta:
            normalized['discriminator_name'] = str(meta[k])
            break
    # other booleans
    for k in ('soft_gumbel_softmax', 'hard_gumbel_softmax', 'batch_discriminator'):
        if k in meta:
            normalized[k] = bool(meta[k])
    return normalized


def load_and_restore_model(run_dir, z_dim=None, force_meta=None):
    """
    Reconstruye el grafo del modelo y restaura los pesos desde el checkpoint en run_dir.
    Usa la mejor información disponible en el orden:
      1) argumentos explícitos en llamada (z_dim)
      2) config.txt (run_sh.sh)
      3) trainer.pkl metadata (si existe)
      4) valores por defecto definidos aquí (coincidentes con run_sh.py)
    Devuelve (data, model, session).
    """
    # dataset (usa la misma ruta utilizada en run_sh.py)
    data = SparseMolecularDataset()
    data.load('data/oe62_9nodes.sparsedataset')

    # 1) leer config.txt
    cfg = parse_config(os.path.join(run_dir, 'config.txt'))

    # 2) intentar metadata desde trainer.pkl
    trainer_meta = load_trainer_metadata(run_dir)

    # 3) decidir valores finales usando prioridades
    final = {}
    # z_dim priority: explicit arg > config.txt > trainer_meta > default 8
    if z_dim is not None:
        final['z_dim'] = int(z_dim)
    elif 'z_dim' in cfg:
        final['z_dim'] = int(cfg['z_dim'])
    elif 'z_dim' in trainer_meta:
        final['z_dim'] = int(trainer_meta['z_dim'])
    else:
        final['z_dim'] = 8

    # decoder_units: trainer_meta > config? (not usually in config.txt) > default
    final['decoder_units'] = trainer_meta.get('decoder_units', (256, 512, 512))

    final['discriminator_units'] = trainer_meta.get('discriminator_units', ((256, 128), 256, (256, 128)))

    # decoder / discriminator selection
    decoder_name = trainer_meta.get('decoder_name', cfg.get('decoder', None))
    discriminator_name = trainer_meta.get('discriminator_name', cfg.get('discriminator', None))

    final['decoder_fn'] = resolve_decoder_decodername(decoder_name) or decoder_rnn
    final['discriminator_fn'] = resolve_discriminator_name(discriminator_name) or encoder_rgcn

    # boolean flags
    final['soft_gumbel_softmax'] = trainer_meta.get('soft_gumbel_softmax', True)
    final['hard_gumbel_softmax'] = trainer_meta.get('hard_gumbel_softmax', False)
    final['batch_discriminator'] = trainer_meta.get('batch_discriminator', True)

    # log what we inferred
    print("Metadata inferida para reconstrucción del modelo:")
    for kk, vv in final.items():
        if kk.endswith('_fn'):
            print(f"  {kk}: {getattr(vv, '__name__', str(vv))}")
        else:
            print(f"  {kk}: {vv}")

    # Reconstruir el modelo con los hiperparámetros decididos
    model = GraphGANModel(data.vertexes,
                          data.bond_num_types,
                          data.atom_num_types,
                          final['z_dim'],
                          decoder_units=final['decoder_units'],
                          discriminator_units=final['discriminator_units'],
                          decoder=final['decoder_fn'],
                          discriminator=final['discriminator_fn'],
                          soft_gumbel_softmax=final['soft_gumbel_softmax'],
                          hard_gumbel_softmax=final['hard_gumbel_softmax'],
                          batch_discriminator=final['batch_discriminator'])

    # Crear optimizador compatible (para restaurar todas las variables si están en el checkpoint)
    optimizer = GraphGANOptimizer(model, learning_rate=1e-4, feature_matching=True)

    # Crear sesión y restaurar
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()  # por defecto toma todas las variables del grafo
    ckpt = tf.train.latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No se encontró checkpoint en {run_dir}. Archivos esperados: model.ckpt.*")
    print(f"Restaurando checkpoint {ckpt} ...")
    saver.restore(session, ckpt)
    print("Checkpoint restaurado OK.")

    # devolver también algunos metadatos por si el llamador quiere guardar info
    return data, model, session, final


def guardar_moleculas_grid(mols, mols_por_fila=5, nombre_archivo="output.png", subImgSize=(200, 200)):
    """Genera y guarda una imagen de moléculas en una grilla como imagen PNG."""
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


def sample_from_run(run_dir, n_samples=100, mols_per_row=10, out_png=None, z_dim=None):
    """
    Carga el modelo de run_dir, genera n_samples moléculas y guarda imagen + devuelve SMILES.
    """
    data, model, session, meta = load_and_restore_model(run_dir, z_dim=z_dim)

    # Generar latentes y muestras
    z = model.sample_z(n_samples)
    mols = samples(data, model, session, z, sample=True)

    # guardar imagen si se pidió
    if out_png:
        guardar_moleculas_grid(mols, mols_por_fila=mols_per_row, nombre_archivo=out_png)

    # obtener smiles
    smiles = mols_to_smiles(mols)
    return mols, smiles, meta


def main():
    parser = argparse.ArgumentParser(description="Sample molecules from a trained run directory.")
    parser.add_argument("--run_dir", required=True, help="Carpeta que contiene checkpoint/model.ckpt.* y config.txt")
    parser.add_argument("--n_samples", type=int, default=100, help="Número de moléculas a muestrear")
    parser.add_argument("--out_png", type=str, default=None, help="Ruta para guardar la imagen de salida (png). Si no, no guarda imagen.")
    parser.add_argument("--mols_per_row", type=int, default=10, help="Moléculas por fila en la imagen")
    parser.add_argument("--print_smiles", action="store_true", help="Imprimir SMILES por consola")
    parser.add_argument("--z_dim", type=int, default=None, help="Forzar z_dim si no hay config/trainer.pkl o quieres sobreescribirla")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No existe la carpeta indicada: {run_dir}")

    mols, smiles, meta = sample_from_run(run_dir, n_samples=args.n_samples, mols_per_row=args.mols_per_row,
                                         out_png=args.out_png, z_dim=args.z_dim)

    if args.print_smiles:
        for i, s in enumerate(smiles):
            print(f"{i+1}\t{s}")

    if args.out_png:
        print(f"Imagen guardada en: {args.out_png}")
    print(f"Se generaron {len([m for m in mols if m is not None])} moléculas válidas de {len(mols)}.")
    # mostrar metadatos inferidos para registro
    print("Metadatos inferidos/empleados para reconstrucción:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
