"""
Surface cleavage energy calculation task.

This task evaluates model performance on predicting surface cleavage energies.

Dataset is retrieved from Ardavan Mehdizadeh and Peter Schindler 2025 AI Sci. 1 025002

Only 10% of the dataset is randomly sampled for testing.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import ast

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from lambench.models.ase_models import ASEModel


def parse_dict_string(d_str):
    """Parse dictionary from string representation."""
    if isinstance(d_str, dict):
        return d_str
    try:
        return json.loads(d_str)
    except json.JSONDecodeError:
        return ast.literal_eval(d_str)


def run_inference(
    model: ASEModel,
    test_data: Path,
) -> dict[str, float]:
    """
    Calculate surface cleavage energies using the provided model.

    Args:
        model: ASEModel instance with loaded calculator
        test_data: Path to directory containing test data JSON file

    Returns:
        Dictionary with MAE and RMSE in meV/A^2
    """
    calc = model.calc

    # Find JSON file in the directory
    json_files = list(test_data.glob("*.json"))
    if not json_files:
        logging.error(f"No JSON file found in {test_data}")
        return {}

    json_file = json_files[0]  # Use the first JSON file found
    logging.info(f"Loading test data from {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    data = json_data.get("data", [])
    adaptor = AseAtomsAdaptor()

    predictions = []
    labels = []

    for item in tqdm(data, desc="Evaluating cleavage energy"):
        try:
            target_cleavage = item.get("cleavage_energy_DFT")

            slab_dict = parse_dict_string(item["structure_slab"])
            bulk_dict = parse_dict_string(item["structure_bulk"])

            struct_slab = Structure.from_dict(slab_dict)
            struct_bulk = Structure.from_dict(bulk_dict)

            atoms_slab = adaptor.get_atoms(struct_slab)
            atoms_bulk = adaptor.get_atoms(struct_bulk)

            # Use calculator
            atoms_slab.calc = calc
            e_slab = atoms_slab.get_potential_energy()

            atoms_bulk.calc = calc
            e_bulk = atoms_bulk.get_potential_energy()

            n_bulk = len(atoms_slab) / len(atoms_bulk)
            area_slab = item.get("area_slab")

            e_cleavage_pred = (e_slab - n_bulk * e_bulk) / (2.0 * area_slab)

            predictions.append(e_cleavage_pred)
            labels.append(target_cleavage)
        except Exception as e:
            logging.warning(f"Failed to calculate cleavage energy for one sample: {e}")
            continue

    if not labels:
        logging.warning("No structures were successfully evaluated.")
        return {}

    return {
        "MAE": mean_absolute_error(labels, predictions) * 1000,  # meV/A^2
        "RMSE": root_mean_squared_error(labels, predictions) * 1000,  # meV/A^2
    }
