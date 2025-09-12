import random
from rdkit import Chem

# Rutas de entrada y salida
sdf_gdb9 = "../data/gdb9_filtrado.sdf"
sdf_oe62 = "../data/OE62_10-20.sdf"
sdf_out = "../data/db_100.sdf"

# Función para leer moléculas de un archivo .sdf
def load_molecules(path):
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    return mols

# Cargar moléculas
print("Cargando moléculas...")
gdb9_mols = load_molecules(sdf_gdb9)
oe62_mols = load_molecules(sdf_oe62)

print(f"Moléculas en gdb9: {len(gdb9_mols)}")
print(f"Moléculas en OE62: {len(oe62_mols)}")

# Seleccionar aleatoriamente
gdb9_sample = random.sample(gdb9_mols, 5000)
oe62_sample = random.sample(oe62_mols, 100)

# Juntar
final_mols = gdb9_sample + oe62_sample

# Guardar en un único archivo SDF
print(f"Guardando {len(final_mols)} moléculas en {sdf_out}...")
writer = Chem.SDWriter(sdf_out)
for mol in final_mols:
    writer.write(mol)
writer.close()

print("Proceso terminado ✅")
