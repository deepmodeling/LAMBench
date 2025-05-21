"""
The Wiggle150 dataset is obtained from the following paper:

@article{doi:10.1021/acs.jctc.5c00015,
author = {Brew, Rebecca R. and Nelson, Ian A. and Binayeva, Meruyert and Nayak, Amlan S. and Simmons, Wyatt J. and Gair, Joseph J. and Wagen, Corin C.},
title = {Wiggle150: Benchmarking Density Functionals and Neural Network Potentials on Highly Strained Conformers},
journal = {Journal of Chemical Theory and Computation},
volume = {21},
number = {8},
pages = {3922-3929},
year = {2025},
doi = {10.1021/acs.jctc.5c00015},
    note ={PMID: 40211427},
URL = {https://doi.org/10.1021/acs.jctc.5c00015},
eprint = {https://doi.org/10.1021/acs.jctc.5c00015}
}
"""

from pathlib import Path
from ase.io import read

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import numpy as np
from lambench.models.ase_models import ASEModel


def run_inference(
    model: ASEModel,
    test_data: Path,
) -> dict[str, float]:
    traj = read(test_data, index=":")
    ado_traj = traj[0:51]
    bpn_traj = traj[51:102]
    efa_traj = traj[102:153]

    preds = []
    label = []

    for traj in [ado_traj, bpn_traj, efa_traj]:
        referen_energy = traj[0].get_potential_energy()
        for i, atoms in enumerate(traj[1:]):
            label_energy = atoms.get_potential_energy()
            label.append(label_energy - referen_energy)
            atoms.calc = model.calc
            try:
                pred_energy = atoms.get_potential_energy()
            except Exception as e:
                print(f"Error in frame {i} of trajectory: {e}")
                pred_energy = np.nan

            preds.append(pred_energy - referen_energy)

    try:
        preds = np.array(preds)
        label = np.array(label)
        mae = mean_absolute_error(label, preds)
        rmse = root_mean_squared_error(label, preds)
    except ValueError:
        mae, rmse = np.nan, np.nan

    return {
        "MAE": mae,  # kcal/mol
        "RMSE": rmse,  # kcal/mol
    }
