import os
import random
from rdkit import Chem
from rdkit.Chem import Draw

def main():
    # Ruta al archivo SDF
    sdf_path = "../data/OE62_filtrado.sdf"

    # Leer moléculas desde el SDF
    suppl = Chem.SDMolSupplier(sdf_path)
    molecules = [mol for mol in suppl if mol is not None]

    if not molecules:
        print("No se pudieron leer moléculas del archivo SDF.")
        return

    # Seleccionar 25 moléculas aleatorias (si hay menos, toma todas)
    num_to_pick = min(25, len(molecules))
    sample_mols = random.sample(molecules, num_to_pick)

    # Crear grid de 5x5
    img = Draw.MolsToGridImage(
        sample_mols,
        molsPerRow=5,
        subImgSize=(700, 700),
        legends=[f"Mol {i+1}" for i in range(num_to_pick)]
    )

    # Mostrar imagen
    img.show()

if __name__ == "__main__":
    main()
