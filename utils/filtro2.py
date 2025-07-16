#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Configuración de filtros
CONFIG = {
    'filter_zero_formal_charge': True,
    'filter_num_atoms': True,
    'min_atoms': 20,
    'max_atoms': 35,
    'filter_allowed_atoms': True,
    'allowed_atoms': ['C', 'N', 'O', 'H', 'P', 'S'],
    'filter_aromatic_atom': True,
    'filter_ring_sizes': True,
    'allowed_ring_sizes': [5, 6],
    'filter_fused_rings': True,
    'filter_aromatic_rings': True,
    'filter_no_linear_alkanes': True
}

# SMARTS para enlaces lineales C–C no cíclicos ni aromáticos
ALKANE_PATTERN = Chem.MolFromSmarts('[C;!R;!a]-[C;!R;!a]')


def load_data(path):
    """
    Carga df_62k.json y retorna listas paralelas de SMILES y bloques XYZ.
    """
    df = pd.read_json(path, orient='split')
    df = df[['canonical_smiles', 'xyz_pbe_relaxed']].dropna(subset=['canonical_smiles', 'xyz_pbe_relaxed'])
    # Limpieza
    df['canonical_smiles'] = df['canonical_smiles'].astype(str).str.strip()
    df['xyz_pbe_relaxed'] = df['xyz_pbe_relaxed'].astype(str).str.strip().replace(r'[\t\r]+', '', regex=True)
    print(f"[DEBUG] Filas con SMILES y XYZ: {len(df)}")
    return df['canonical_smiles'].tolist(), df['xyz_pbe_relaxed'].tolist()


def embed_xyz_on_mol(mol, xyz_block):
    """
    Asigna las coordenadas de xyz_block a un conformer de mol creado a partir de SMILES.
    """
    lines = xyz_block.splitlines()
    try:
        n = int(lines[0])
        coords = []
        for line in lines[2:2 + n]:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                coords.append((x, y, z))
        if len(coords) != mol.GetNumAtoms():
            AllChem.EmbedMolecule(mol, randomSeed=42)
        else:
            conf = Chem.Conformer(len(coords))
            for i, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(i, Chem.Geometry.Point3D(x, y, z))
            mol.AddConformer(conf, assignId=True)
    except Exception:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol

# Filtros
def filter_zero_formal_charge(mols): return [m for m in mols if all(a.GetFormalCharge() == 0 for a in m.GetAtoms())]

def filter_num_atoms(mols, lo, hi): return [m for m in mols if lo <= m.GetNumAtoms() <= hi]

def filter_allowed_atoms(mols, allowed): return [m for m in mols if all(a.GetSymbol() in allowed for a in m.GetAtoms())]

def filter_aromatic_atom(mols): return [m for m in mols if any(a.GetIsAromatic() for a in m.GetAtoms())]

def filter_ring_sizes(mols, sizes):
    out = []
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        if rings and all(len(r) in sizes for r in rings):
            out.append(m)
    return out

def filter_fused_rings(mols):
    out = []
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        if len(rings) <= 1:
            out.append(m)
        else:
            tot = sum(len(r) for r in rings)
            uniq = len({idx for r in rings for idx in r})
            if tot > uniq:
                out.append(m)
    return out

def filter_aromatic_rings(mols):
    out = []
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        if not rings:
            continue
        ok = True
        for ring in rings:
            if not all(m.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                ok = False
                break
            for i in range(len(ring)):
                b = m.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
                if not b.GetIsAromatic():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(m)
    return out

def filter_no_linear_alkanes(mols): return [m for m in mols if not m.HasSubstructMatch(ALKANE_PATTERN)]

class XYZFilteredDataset:
    def __init__(self, smiles_list, xyz_list):
        self.raw_smiles = smiles_list
        self.raw_xyz = xyz_list
        self.mols = []
        for smi, xyz in zip(smiles_list, xyz_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol = Chem.AddHs(mol)
                mol = embed_xyz_on_mol(mol, xyz)
                Chem.SanitizeMol(mol)
                self.mols.append(mol)
        print(f"[DEBUG] Moléculas tras carga inicial: {len(self.mols)}")
        self.xyz_blocks = []

    def apply_filters(self, cfg):
        steps = [
            ('zero formal charge', filter_zero_formal_charge),
            ('num atoms', lambda ms: filter_num_atoms(ms, cfg['min_atoms'], cfg['max_atoms'])),
            ('allowed atoms', lambda ms: filter_allowed_atoms(ms, cfg['allowed_atoms'])),
            ('aromatic atom', filter_aromatic_atom),
            ('ring sizes', lambda ms: filter_ring_sizes(ms, cfg['allowed_ring_sizes'])),
            ('fused rings', filter_fused_rings),
            ('aromatic rings', filter_aromatic_rings),
            ('no linear alkanes', filter_no_linear_alkanes)
        ]
        for name, func in steps:
            before = len(self.mols)
            key = f"filter_{name.replace(' ', '_')}"
            if cfg.get(key, False):
                self.mols = func(self.mols)
                after = len(self.mols)
                print(f"[DEBUG] Filtro {name}: {before} -> {after}")

    def get_xyz(self):
        out = []
        for m in self.mols:
            conf = m.GetConformer()
            n = m.GetNumAtoms()
            lines = [str(n), 'Filtered XYZ from RDKit']
            for i in range(n):
                a = m.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                lines.append(f"{a.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
            out.append("\n".join(lines))
        self.xyz_blocks = out
        return out

    def save(self, fn):
        data = {
            'smiles': self.raw_smiles,
            'raw_xyz': self.raw_xyz,
            'mols': self.mols,
            'xyz_blocks': self.xyz_blocks
        }
        with open(fn, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] Guardado: {fn}")

if __name__ == '__main__':
    base_dir = '/home/silkenmocha/Documentos/MolGAN-OE62/data/m1507656'
    json_file = os.path.join(base_dir, 'df_62k.json')
    output_file = os.path.join(base_dir, 'OE62_xyz_filtered.pkl')

    smiles, xyzs = load_data(json_file)
    dataset = XYZFilteredDataset(smiles, xyzs)
    dataset.apply_filters(CONFIG)
    dataset.get_xyz()
    dataset.save(output_file)
    print(f"Total final moléculas: {len(dataset.mols)}")
