# ruff: noqa: E402
"""
The test data is obtained from the following paper:

Antoine Loew et al 2026 J. Phys. Mater. 9 015010 Universal machine learning potentials under pressure
DOI 10.1088/2515-7639/ae2ba8

We downsampled the original test set to 45 structures at each pressure point (25, 50, 75, 100, 125, 150 GPa)
"""

from ase.io import read
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from lambench.models.ase_models import ASEModel
from collections import defaultdict
import logging

KBAR_2_EVA3 = 6.2415e-4
GPA_2_KBAR = 10


def optimize(structure: Atoms, target_p: float, fmax: float, steps: int) -> Atoms:
    target_p = target_p * GPA_2_KBAR * KBAR_2_EVA3  # to eV/A3
    cell_filter = FrechetCellFilter(structure, scalar_pressure=target_p)
    opt = FIRE(cell_filter)
    opt.run(fmax=fmax, steps=steps)
    return cell_filter.atoms


def test_one(
    init: Atoms,
    final: Atoms,
    target_p: float,
    calc: Calculator,
    fmax: float,
    max_steps: int,
) -> tuple[float, float]:
    init.calc = calc
    optimized = optimize(init, int(target_p), fmax, max_steps)
    natoms = len(init)
    return final.get_volume() / natoms, optimized.get_volume() / natoms


def run_inference(
    model: ASEModel,
    test_data: Path,
    fmax: float,
    max_steps: int,
) -> dict[str, float]:
    calc = model.calc
    final_res = defaultdict(list)
    num_samples = 0
    num_fails = 0

    for pressure in tqdm(["025", "050", "075", "100", "125", "150"]):
        init_traj = read(f"{test_data}/P{pressure}.traj", ":")
        final_traj = read(f"{test_data}/P{pressure}.traj", ":")
        for i in tqdm(range(len(init_traj))):
            init = init_traj[i]
            final = final_traj[i]
            assert init.get_chemical_formula() == final.get_chemical_formula()
            try:
                dft, lam = test_one(init, final, int(pressure), calc, fmax, max_steps)
            except Exception as e:
                logging.error(
                    f"Error during test_one at pressure {pressure}, index {i}: {e}"
                )
                dft, lam = None, None
            if dft is None or lam is None:
                num_fails += 1
                continue
            num_samples += 1
            final_res[f"{pressure}_labels"].append(dft)
            final_res[f"{pressure}_preds"].append(lam)

    return {
        "MAE": mean_absolute_error(
            final_res[f"{pressure}_labels"], final_res[f"{pressure}_preds"]
        ),  # A3/atom
        "RMSE": root_mean_squared_error(
            final_res[f"{pressure}_labels"], final_res[f"{pressure}_preds"]
        ),  # A3/atom
        "success_rate": (num_samples - num_fails) / num_samples,
    }
