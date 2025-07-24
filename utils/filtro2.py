import os
import pickle
import pandas as pd
from rdkit import Chem

# Configuration: activate/deactivate filters and set parameters
CONFIG = {
    'filter_zero_formal_charge': False,
    'filter_num_atoms': True,
    'min_atoms': 1,
    'max_atoms': 20,
    'filter_allowed_atoms': True,  # keep only molecules with these atoms
    'allowed_atoms': ['C', 'N', 'O', 'H', 'P', 'S'],
    'filter_aromatic_atom': False,
    'filter_ring_sizes': True,
    'allowed_ring_sizes': [5, 6],  # e.g., [5, 6]
    'filter_fused_rings': False,
    'filter_aromatic_rings': True,
    'filter_no_linear_alkanes': False
}

# SMARTS pattern for linear C–C alkane bonds
ALKANE_PATTERN = Chem.MolFromSmarts('[C;!R;!a]-[C;!R;!a]')


def load_smiles_from_json(path):
    """
    Carga SMILES desde un JSON con orient='split', renombra 'canonical_smiles' a 'smiles',
    limpia valores nulos y caracteres extraños, y devuelve una lista de SMILES.
    """
    # Leer todo el JSON
    df = pd.read_json(path, orient='split')
    # Seleccionar solo la columna necesaria
    df = df[['canonical_smiles']].rename(columns={'canonical_smiles': 'smiles'})
    # Limpiar texto
    df['smiles'] = (
        df['smiles']
        .astype(str)
        .str.strip()
        .str.replace(r'[\t\n\r]+', '', regex=True)
    )
    # Eliminar filas vacías o nulas
    df = df[df['smiles'].notna() & (df['smiles'] != '')]
    return df['smiles'].tolist()


def to_molecules(smiles_list):
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
    return mols

# [Filters definitions remain identical]

def filter_zero_formal_charge(mols):
    return [m for m in mols if all(atom.GetFormalCharge() == 0 for atom in m.GetAtoms())]

def filter_num_atoms(mols, min_atoms, max_atoms):
    return [m for m in mols if min_atoms < m.GetNumAtoms() < max_atoms]

def filter_allowed_atoms(mols, allowed):
    return [m for m in mols if all(atom.GetSymbol() in allowed for atom in m.GetAtoms())]

def filter_aromatic_atom(mols):
    return [m for m in mols if any(atom.GetIsAromatic() for atom in m.GetAtoms())]

def filter_ring_sizes(mols, allowed_sizes):
    filtered = []
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        if rings and all(len(r) in allowed_sizes for r in rings):
            filtered.append(m)
    return filtered


def filter_fused_rings(mols):
    filtered = []
    for m in mols:
        atom_rings = m.GetRingInfo().AtomRings()
        if len(atom_rings) <= 1:
            filtered.append(m)
            continue
        total = sum(len(r) for r in atom_rings)
        unique = len({atom for ring in atom_rings for atom in ring})
        if total > unique:
            filtered.append(m)
    return filtered


def filter_aromatic_rings(mols):
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
        self.smiles = None

    def apply_filters(self, config):
        if config['filter_zero_formal_charge']:
            self.mols = filter_zero_formal_charge(self.mols)
        if config['filter_num_atoms']:
            self.mols = filter_num_atoms(self.mols, config['min_atoms'], config['max_atoms'])
        if config['filter_allowed_atoms']:
            self.mols = filter_allowed_atoms(self.mols, config['allowed_atoms'])
        if config['filter_aromatic_atom']:
            self.mols = filter_aromatic_atom(self.mols)
        if config['filter_ring_sizes']:
            self.mols = filter_ring_sizes(self.mols, config['allowed_ring_sizes'])
        if config['filter_fused_rings']:
            self.mols = filter_fused_rings(self.mols)
        if config['filter_aromatic_rings']:
            self.mols = filter_aromatic_rings(self.mols)
        if config['filter_no_linear_alkanes']:
            self.mols = filter_no_linear_alkanes(self.mols)

    def get_smiles(self):
        return [Chem.MolToSmiles(m) for m in self.mols]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)


if __name__ == '__main__':
    # Paths (estás en utils, así que ../data sube al directorio correcto)
    base_dir = '../data'
    input_file = os.path.join(base_dir, 'm1507656', 'df_62k.json')
    output_file = os.path.join(base_dir, 'OE62_2.smi')  # Guardado en la misma carpeta data/

    # Load
    smiles = load_smiles_from_json(input_file)

    # Filter
    dataset = FilteredDataset(smiles)
    dataset.apply_filters(CONFIG)

    # Retrieve SMILES from filtered Mol objects
    dataset.smiles = dataset.get_smiles()

    # Save SMILES to .smi file (1 SMILES por línea)
    with open(output_file, 'w') as f:
        for smi in dataset.smiles:
            f.write(smi + '\n')

    print(f"Filtered dataset saved to {output_file}, total molecules: {len(dataset.smiles)}")

