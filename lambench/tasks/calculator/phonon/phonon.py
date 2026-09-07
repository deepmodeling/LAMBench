"""
Code adapted from the following paper and code:

@misc{loew2024universalmachinelearninginteratomic,
      title={Universal Machine Learning Interatomic Potentials are Ready for Phonons},
      author={Antoine Loew and Dewen Sun and Hai-Chen Wang and Silvana Botti and Miguel A. L. Marques},
      year={2024},
      eprint={2412.16551},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2412.16551},
}
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import phonopy
import yaml
from ase import Atoms
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from lambench.models.ase_models import ASEModel
from lambench.tasks.calculator.phonon.phonon_utils import (
    THz_TO_K,
    ase_to_phonopy_atoms,
    force_observer,
    phonopy_to_ase_atoms,
)


def run_phonon_simulation_single(
    model: ASEModel,
    phonon_file: Path,
    distance: float,
    workdir: Path,
) -> dict[str, float] | None:
    """
    Run phonon related calculations for a single given phonon file.

    Parameters:
        model: ASEModel object.
        phonon_file: Path to the phonon file.
        distance: Distance for displacements.
        workdir: Path to the working directory.
    """
    try:
        # Step 1: Run relaxation
        atoms: Atoms = phonopy_to_ase_atoms(phonon_file)
        atoms = model.run_ase_relaxation(
            atoms, model.calc, fmax=1e-3, observer=force_observer
        )

        # Step 2: Convert ASE Atoms object to PhonopyAtoms object
        phonon_atoms = ase_to_phonopy_atoms(atoms)
        phonon = phonopy.Phonopy(
            phonon_atoms,
            supercell_matrix=atoms.info["supercell_matrix"],
            primitive_matrix=atoms.info["primitive_matrix"],
        )

        # Step 3: Generate displacements
        phonon.generate_displacements(distance=distance, is_diagonal=False)

        # Step 4: Calculate force constants
        forcesets = []

        for frame in phonon.supercells_with_displacements:
            frame_atom = Atoms(
                cell=frame.cell,
                symbols=frame.symbols,
                scaled_positions=frame.scaled_positions,
                pbc=True,
            )
            frame_atom.calc = model.calc
            forces = frame_atom.get_forces()
            drift_force = forces.sum(axis=0)
            for force in forces:
                force -= drift_force / forces.shape[0]
            forcesets.append(forces)

        phonon.forces = forcesets
        phonon.produce_force_constants()
        phonon.symmetrize_force_constants()

        # Step 5: save output files

        phonon.save(workdir / phonon_file.name, settings={"force_constants": True})

        # Step 6: Calculate thermal properties
        phonon.init_mesh()
        phonon.run_mesh()
        phonon.run_thermal_properties(temperatures=(300,))
        thermal_dict = phonon.get_thermal_properties_dict()

        commensurate_q = get_commensurate_points(phonon.supercell_matrix)
        phonon_freqs = np.array([phonon.get_frequencies(q) for q in commensurate_q])

        # Step 7: Updata output files
        with open(workdir / phonon_file.name, "r") as f:
            output = yaml.load(f, yaml.FullLoader)

        output["free_e"] = thermal_dict["free_energy"].tolist()
        output["entropy"] = thermal_dict["entropy"].tolist()
        output["heat_capacity"] = thermal_dict["heat_capacity"].tolist()
        output["phonon_freq"] = phonon_freqs.tolist()

        # TODO: optional: update and save output files
        return {
            "mp_id": phonon_file.name.split(".")[0],
            "entropy": output["entropy"][0],
            "heat_capacity": output["heat_capacity"][0],
            "free_energy": output["free_e"][0],
            "max_freq": np.max(np.array(phonon_freqs)) * THz_TO_K,
        }

    except Exception as e:
        logging.error(f"Error occured for {phonon_file.name!s}: {e}")
        return None


def run_phonon_simulation(
    model: ASEModel,
    test_data: Path,
    distance: float,
    workdir: Path,
) -> dict[str, float]:
    """
    This function runs phonon simulations for a list of test systems using the given model.
    """
    test_files = list(test_data.glob("*.yaml.bz2"))
    if len(test_files) == 0:
        logging.error("No test files found.")
        return {}
    logging.info(f"Running phonon simulations for {len(test_files)} files...")

    dataframe_rows = []
    for test_file in tqdm(test_files):
        result = run_phonon_simulation_single(
            model,
            test_file,
            distance,
            workdir,
        )
        logging.info(f"Simulation completed for system {test_file.name!s}.\n")

        if result is not None:
            dataframe_rows.append(result)
    preds = pd.DataFrame(dataframe_rows)
    preds.to_csv(workdir / "phonon-preds.csv", index=False)

    # Post-processing
    results = {}
    try:
        labels = pd.read_csv(test_data / "pbe.csv")
        TOTAL_RECORDS = len(labels)
        preds.sort_values("mp_id", inplace=True)
        labels.sort_values("mp_id", inplace=True)

        # Filter predictions and labels based on valid mp_ids
        valid_preds = preds[
            np.isfinite(preds[["free_energy", "heat_capacity"]]).all(axis=1)
        ]
        valid_mp_ids = set(valid_preds["mp_id"])
        labels = labels[labels["mp_id"].isin(valid_mp_ids)]
        preds = valid_preds

        success_rate = len(preds) / TOTAL_RECORDS
        mae_wmax = mean_absolute_error(labels["max_freq"], preds["max_freq"])
        mae_s = mean_absolute_error(labels["entropy"], preds["entropy"])
        mae_f = mean_absolute_error(labels["free_energy"], preds["free_energy"])
        mae_c = mean_absolute_error(labels["heat_capacity"], preds["heat_capacity"])
        results = {
            "success_rate": success_rate,
            "mae_max_freq": mae_wmax,
            "mae_entropy": mae_s,
            "mae_free_energy": mae_f,
            "mae_heat_capacity": mae_c,
        }
    except Exception as e:
        logging.error(f"Error occured during post-processing: {e}")
    return results
