"""
The test data is retrieved from:
Chem. Mater. 2023, 35, 24, 10619–10634

https://pubs.acs.org/doi/10.1021/acs.chemmater.3c02251

Only 1813 structure pairs are used.

"""

from ase.io import read
import numpy as np
from ase import Atoms
from tqdm import tqdm
from pathlib import Path
from ase.io import read

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from lambench.models.ase_models import ASEModel
import logging



def get_oxygen_reference_energy(calc) -> float:

    vacuum_size = 30      # Ångströms: Large cell size to ensure vacuum separation
    o_o_bond_length = 1.23  # Ångströms: Experimental O-O bond length for O2
    cell_vector = vacuum_size
    cell = [cell_vector, cell_vector, cell_vector]
    center = cell_vector / 2

    positions = [
        (center, center, center - o_o_bond_length / 2),
        (center, center, center + o_o_bond_length / 2)
    ]

    molecular_oxygen = Atoms(
        'O2',
        positions=positions,
        cell=cell,
        pbc=True
    )
    molecular_oxygen.calc = calc
    return molecular_oxygen.get_potential_energy()/2

def run_inference(
    model: ASEModel,
    test_data: Path,
) -> dict[str, float]:
    pristine_structures = read(test_data / "vacancy_pristine_structures.traj", ":")
    defect_structures = read(test_data / "vacancy_defect_structures.traj", ":")
    labels = np.load(test_data / "vacancy_evf_label.npy")


    evf_lab = []
    evf_pred = []
    calc = model.calc

    # Calculate reference energy for oxygen atom
    E_o = get_oxygen_reference_energy(calc)

    for pristine, defect, label in tqdm(zip(pristine_structures, defect_structures, labels)):

        natoms_pri = len(pristine)
        natoms_def = len(defect)
        
        n_oxygen = natoms_pri - natoms_def

        pristine.calc = calc
        defect.calc = calc
        try:
            final = defect.get_potential_energy()
            initial = pristine.get_potential_energy()

            e_vf = final + n_oxygen * E_o - initial
            evf_lab.append(label)
            evf_pred.append(e_vf)

        except Exception as e:
            logging.error(f"Error occurred while processing structures: {e}")
        
    return {
        "MAE": mean_absolute_error(evf_lab, evf_pred),  # eV
        "RMSE": root_mean_squared_error(evf_lab, evf_pred),  # eV
    }
