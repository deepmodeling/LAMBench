"""
The test data is retrieved from:
J. Chem. Inf. Model. 2020, 60, 3, 1453–1460

https://pubs.acs.org/doi/10.1021/acs.jcim.9b01171

Only the PLF547 dataset is used.

"""

from ase.io import read
import numpy as np
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from pathlib import Path
from lambench.models.ase_models import ASEModel
import logging


def run_inference(
    model: ASEModel,
    test_data: Path,
) -> dict[str, float]:
    active_site_atoms = read(test_data / "active_site.traj", ":")
    drug_atoms = read(test_data / "drug.traj", ":")
    combined_atoms = read(test_data / "combined.traj", ":")
    labels = np.load(test_data / "labels.npy")

    EV_TO_KCAL = 23.06092234465

    calc = model.calc
    preds = []
    success_labels = []

    for site, drug, combo, label in tqdm(
        zip(active_site_atoms, drug_atoms, combined_atoms, labels)
    ):
        try:
            for atoms in (site, drug, combo):
                atoms.calc = calc
                atoms.info.update(
                    {"fparam": np.array([atoms.info["charge"], atoms.info["spin"]])}
                )

            site_energy = site.get_potential_energy()
            drug_energy = drug.get_potential_energy()
            combo_energy = combo.get_potential_energy()

            binding_energy = combo_energy - site_energy - drug_energy
            preds.append(binding_energy * EV_TO_KCAL)
            success_labels.append(label)
        except Exception as e:
            logging.warning(f"Failed to calculate binding energy for one sample: {e}")
            continue

    return {
        "MAE": mean_absolute_error(success_labels, preds),  # kcal/mol
        "RMSE": root_mean_squared_error(success_labels, preds),  # kcal/mol
        "success_rate": len(success_labels) / len(labels),
    }
