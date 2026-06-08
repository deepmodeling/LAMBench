"""
RXN-Path-39: 13 organic reactions (wB97M-V/def2-TZVPD), 3 path-sampling
trajectories each, 11 arc-length-equidistant frames per trajectory.

For each trajectory the first frame (index 0) is chosen as the reference.
The task measures how accurately a LAM reproduces the relative energies of all
other frames with respect to that reference, i.e.

    ΔE_DFT(i)  = E_DFT(i)  − E_DFT(frame 0)   [kcal/mol]
    ΔE_LAM(i)  = E_LAM(i)  − E_LAM(frame 0)   [kcal/mol]

and reports MAE and RMSE over all 39 × 10 = 390 (reaction, frame) pairs.
"""

from pathlib import Path
import logging

import numpy as np
from ase.io import Trajectory
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from lambench.models.ase_models import ASEModel

EV_TO_KCAL = 23.0609  # 1 eV = 23.0609 kcal/mol


def run_inference(model: ASEModel, test_data: Path) -> dict[str, float]:
    """
    Parameters
    ----------
    model : ASEModel
    test_data : Path
        Root of the trajectory tree.  Expected layout::

            test_data/
              <reaction_id>/
                traj_0.traj
                traj_1.traj
                traj_2.traj
              ...

    Returns
    -------
    dict with keys "MAE" and "RMSE" in kcal/mol.
    """
    calc = model.calc
    label_diffs: list[float] = []
    pred_diffs: list[float] = []

    traj_files = sorted(test_data.rglob("traj_*.traj"))
    if not traj_files:
        raise FileNotFoundError(f"No traj_*.traj files found under {test_data}")

    for traj_path in traj_files:
        frames = list(Trajectory(traj_path))

        # DFT reference energies (eV, stored by SinglePointCalculator)
        dft_energies = np.array([a.get_potential_energy() for a in frames])
        ref_dft_kcal = dft_energies[0] * EV_TO_KCAL

        # LAM energy for the first frame (reference)
        frames[0].calc = calc
        try:
            ref_pred_kcal = frames[0].get_potential_energy() * EV_TO_KCAL
        except Exception as e:
            logging.error(
                f"Failed predicting reference frame (idx=0) in {traj_path}: {e}"
            )
            continue  # skip this trajectory entirely

        # Relative energies for every non-reference frame
        for i, atoms in enumerate(frames):
            if i == 0:
                continue

            label_diffs.append(dft_energies[i] * EV_TO_KCAL - ref_dft_kcal)

            atoms.calc = calc
            try:
                pred_kcal = atoms.get_potential_energy() * EV_TO_KCAL
            except Exception as e:
                logging.error(
                    f"Failed predicting frame {i} of {traj_path}: {e}"
                )
                pred_kcal = np.nan
            pred_diffs.append(pred_kcal - ref_pred_kcal)

    label_arr = np.array(label_diffs)
    pred_arr = np.array(pred_diffs)
    valid = np.isfinite(pred_arr)

    if not valid.any():
        logging.error("All predictions failed; returning NaN metrics.")
        return {"MAE": np.nan, "RMSE": np.nan}

    if not valid.all():
        n_failed = int((~valid).sum())
        logging.warning(
            f"{n_failed} frame(s) failed inference and were excluded from metrics."
        )

    return {
        "MAE": float(mean_absolute_error(label_arr[valid], pred_arr[valid])),
        "RMSE": float(root_mean_squared_error(label_arr[valid], pred_arr[valid])),
    }
