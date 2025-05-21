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
import logging

EV_TO_KCAL = 23.0609  # eV to kcal/mol


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

    for sub_traj in [ado_traj, bpn_traj, efa_traj]:
        ref_energy_label = sub_traj[0].get_potential_energy()
        sub_traj[0].calc = model.calc
        try:
            ref_energy_pred = sub_traj[0].get_potential_energy() * EV_TO_KCAL
        except Exception:
            continue
        for i, atoms in enumerate(sub_traj[1:]):
            label_energy = atoms.get_potential_energy()
            label.append(label_energy - ref_energy_label)
            atoms.calc = model.calc
            try:
                pred_energy = atoms.get_potential_energy() * EV_TO_KCAL
            except Exception as e:
                logging.error(f"Error in frame {i} of trajectory: {e}")
                pred_energy = np.nan

            preds.append(pred_energy - ref_energy_pred)

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
