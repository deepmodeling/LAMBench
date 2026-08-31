from pathlib import Path

import phonopy
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms

# Constants unit conversion
THz_TO_K = 47.9924


def ase_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    """
    Convert ASE Atoms object to PhonopyAtoms object.
    """
    # Extract atomic symbols and positions
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    masses = atoms.get_masses()

    return PhonopyAtoms(symbols=symbols, positions=positions, cell=cell, masses=masses)


def phonopy_to_ase_atoms(phonon_file: Path) -> Atoms:
    """
    Convert PhonopyAtoms object to ASE Atoms object.
    """
    phonon = phonopy.load(phonon_file)
    return Atoms(
        cell=phonon.unitcell.cell,
        symbols=phonon.unitcell.symbols,
        scaled_positions=phonon.unitcell.scaled_positions,
        pbc=True,
        info={
            "supercell_matrix": phonon.supercell_matrix,
            "primitive_matrix": phonon.primitive_matrix,
        },
    )


def force_observer(atoms: Atoms) -> None:
    """
    Check if the forces are physical.
    """
    fsqr_max = (atoms.get_forces() ** 2).sum(axis=1).max()
    if fsqr_max > 10000.0**2:
        raise Exception("Error forces are unphysical")
