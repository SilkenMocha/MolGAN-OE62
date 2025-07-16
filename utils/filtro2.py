import os
import pickle
import pandas as pd
from rdkit import Chem

# Configuration: activate/deactivate filters and set parameters
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

# SMARTS pattern for linear C–C alkane bonds
ALKANE_PATTERN = Chem.MolFromSmarts('[C;!R;!a]-[C;!R;!a]')


def load_smiles_from_json(path):
    df = pd.read_json(path, orient='split')
    df = df[['canonical_smiles']].rename(columns={'canonical_smiles': 'smiles'})
    df['smiles'] = (
        df['smiles']
        .astype(str)
        .str.strip()
        .str.replace(r'[\t\n\r]+', '', regex=True)
    )
    df = df[df['smiles'].notna() & (df['smiles'] != '')]
    return df['smiles'].tolist()


def to_molecules(smiles_list):
    return [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]


def filter_zero_formal_charge(mols):
    return [m for m in mols if all(atom.GetFormalCharge() == 0 for atom in m.GetAtoms())]

def filter_num_atoms(mols, min_atoms, max_atoms):
    return [m for m in mols if min_atoms < m.GetNumAtoms() < max_atoms]

def filter_allowed_atoms(mols, allowed):
    return [m for m in mols if all(atom.GetSymbol() in allowed for atom in m.GetAtoms())]

def filter_aromatic_atom(mols):
    return [m for m in mols if any(atom.GetIsAromatic() for atom in m.GetAtoms())]

def filter_ring_sizes(mols, allowed_sizes):
    return [m for m in mols if m.GetRingInfo().AtomRings() and all(len(r) in allowed_sizes for r in m.GetRingInfo().AtomRings())]

def filter_fused_rings(mols):
    filtered = []
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        if len(rings) <= 1:
            filtered.append(m)
        else:
            total = sum(len(r) for r in rings)
            unique = len({a for ring in rings for a in ring})
            if total > unique:
                filtered.append(m)
    return filtered

def filter_aromatic_rings(mols):
    """
    Conserva solo moléculas cuyas **todas** los anillos sean completamente aromáticos.
    """
    filtered = []
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        # Descarta moléculas sin anillos
        if not rings:
            continue
        # Verificar que todos los anillos sean aromáticos en átomos y enlaces
        all_rings_aromatic = True
        for ring in rings:
            # todos los átomos aromáticos
            if not all(m.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                all_rings_aromatic = False
                break
            # todos los enlaces aromáticos
            for i in range(len(ring)):
                a1, a2 = ring[i], ring[(i+1) % len(ring)]
                bond = m.GetBondBetweenAtoms(a1, a2)
                if not bond.GetIsAromatic():
                    all_rings_aromatic = False
                    break
            if not all_rings_aromatic:
                break
        if all_rings_aromatic:
            filtered.append(m)
    return filtered

def filter_no_linear_alkanes(mols):
    return [m for m in mols if not m.HasSubstructMatch(ALKANE_PATTERN)]


class FilteredDataset:
    def __init__(self, smiles_list):
        self.mols = to_molecules(smiles_list)
        print(f"[DEBUG] Molecules loaded: {len(self.mols)}")

    def apply_filters(self, config):
        # List of filters in order applied, with names
        filters = [
            ('zero formal charge', filter_zero_formal_charge, 'filter_zero_formal_charge'),
            ('number of atoms', lambda ms: filter_num_atoms(ms, config['min_atoms'], config['max_atoms']), 'filter_num_atoms'),
            ('allowed atoms', lambda ms: filter_allowed_atoms(ms, config['allowed_atoms']), 'filter_allowed_atoms'),
            ('at least one aromatic atom', filter_aromatic_atom, 'filter_aromatic_atom'),
            ('ring sizes', lambda ms: filter_ring_sizes(ms, config['allowed_ring_sizes']), 'filter_ring_sizes'),
            ('fused rings', filter_fused_rings, 'filter_fused_rings'),
            ('aromatic rings', filter_aromatic_rings, 'filter_aromatic_rings'),
            ('no linear alkanes', filter_no_linear_alkanes, 'filter_no_linear_alkanes')
        ]
        for name, func, key in filters:
            if config.get(key):
                before = len(self.mols)
                self.mols = func(self.mols)
                after = len(self.mols)
                print(f"[DEBUG] After '{name}' filter: {before} -> {after}")
        return self.mols

    def get_smiles(self):
        return [Chem.MolToSmiles(m) for m in self.mols]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'smiles': self.get_smiles()}, f)


if __name__ == '__main__':
    # Paths
    base_dir = '/home/silkenmocha/Documentos/MolGAN-OE62/data/m1507656'
    input_file = os.path.join(base_dir, 'df_62k.json')
    output_file = os.path.join(base_dir, 'OE62.txt')

    # Load
    smiles = load_smiles_from_json(input_file)

    # Filter
    dataset = FilteredDataset(smiles)
    dataset.apply_filters(CONFIG)

    # Retrieve SMILES and overwrite
    dataset.smiles = dataset.get_smiles()

    # Save
    dataset.save(output_file)
    print(f"Filtered dataset saved to {output_file}, total molecules: {len(dataset.smiles)}")
    base_dir = '/home/silkenmocha/Documentos/MolGAN-OE62/data/m1507656'
    json_file = os.path.join(base_dir, 'df_62k.json')
    out_pickle = os.path.join(base_dir, 'OE62.txt')

    smiles = load_smiles_from_json(json_file)
    ds = FilteredDataset(smiles)
    ds.apply_filters(CONFIG)
    ds.save(out_pickle)
    print(f"Guardadas {len(ds.get_smiles())} SMILES en {out_pickle}.")
