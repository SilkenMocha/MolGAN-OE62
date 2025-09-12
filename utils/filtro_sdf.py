import os
from rdkit import Chem

# Configuración de filtros
CONFIG = {
    'filter_zero_formal_charge': False,
    'filter_num_atoms': True,
    'min_atoms': 10,
    'max_atoms': 20,
    'filter_allowed_atoms': True,
    'allowed_atoms': ['C', 'N', 'O', 'H', 'P', 'S'],
    'filter_aromatic_atom': False,
    'filter_ring_sizes': True,
    'allowed_ring_sizes': [5, 6],
    'filter_fused_rings': False,
    'filter_aromatic_rings': True,
    'filter_no_linear_alkanes': False
}

# SMARTS para enlaces lineales C–C (alcano)
ALKANE_PATTERN = Chem.MolFromSmarts('[C;!R;!a]-[C;!R;!a]')


def load_molecules_from_sdf(path):
    """Carga moléculas desde un archivo SDF."""
    suppl = Chem.SDMolSupplier(path)
    return [mol for mol in suppl if mol is not None]


# === Filtros ===
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
        if not rings:
            continue
        all_rings_aromatic = True
        for ring in rings:
            if not all(m.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                all_rings_aromatic = False
                break
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
    def __init__(self, mols):
        self.mols = mols

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


if __name__ == '__main__':
    # Paths
    base_dir = '../data'
    input_file = os.path.join(base_dir, 'OE62.sdf')
    output_file = os.path.join(base_dir, 'OE62_10-20.sdf')

    # Load molecules
    mols = load_molecules_from_sdf(input_file)

    # Apply filters
    dataset = FilteredDataset(mols)
    dataset.apply_filters(CONFIG)

    # Save filtered molecules a un nuevo SDF
    writer = Chem.SDWriter(output_file)
    for mol in dataset.mols:
        writer.write(mol)
    writer.close()

    print(f"Archivo filtrado guardado en {output_file}, total moléculas: {len(dataset.mols)}")
