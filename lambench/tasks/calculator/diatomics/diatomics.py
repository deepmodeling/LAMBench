"""
The reference data (diatomics.json) is derived from the MLIP Arena project:

    MLIP Arena — Benchmark machine learning interatomic potential at scale
    Yuan Chiang, Lawrence Berkeley National Laboratory
    https://github.com/atomind-ai/mlip-arena

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License.  You may obtain a copy of
the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License.

----

Homonuclear diatomics dissociation curves (Applicability).

Per-molecule metric, then arithmetic mean over molecules:

    roughness – RMSE of d²(E_model - E_DFT)/dr²  (eV/Å²)

Leaderboard (Applicability-Roughness ↓) uses avg_roughness.

The last point of every scan is dropped: some PBE dissociation tails have a
spurious endpoint jump, and trimming all scans the same way avoids per-molecule
special cases.

Reference data: lambench/tasks/calculator/diatomics/diatomics.json
    name, method, R, E, F, S^2
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

if TYPE_CHECKING:
    from lambench.models.ase_models import ASEModel

_LABEL_FILE = Path(__file__).parent / "diatomics.json"
_DROP_LAST_POINTS = 1


def _element_from_name(mol_name: str) -> str:
    """Extract element symbol: 'AlAl' → 'Al', 'HH' → 'H'."""
    if len(mol_name) % 2:
        raise ValueError(f"Expected a repeated homonuclear label, got {mol_name!r}")
    half = mol_name[: len(mol_name) // 2]
    if mol_name != half + half:
        raise ValueError(f"Expected a repeated homonuclear label, got {mol_name!r}")
    return half


def _scan_arrays(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Bond lengths (Å) and DFT energies (eV), with the last point dropped."""
    bond_lengths = np.asarray(entry["R"], dtype=float)[:-_DROP_LAST_POINTS]
    dft_energies = np.asarray(entry["E"], dtype=float)[:-_DROP_LAST_POINTS]
    if bond_lengths.size != dft_energies.size:
        raise ValueError(
            f"{entry.get('name', '<unknown>')}: R and E length mismatch after trim"
        )
    return bond_lengths, dft_energies


def _compute_roughness(residuals: np.ndarray, dr: float) -> float | None:
    """RMSE of d² residual / dr².  None if too few finite points."""
    delta2 = np.diff(residuals, n=2)
    valid = delta2[np.isfinite(delta2)]
    if len(valid) == 0:
        return None
    return float(np.sqrt(np.mean((valid / dr**2) ** 2)))


def _predict_energies(
    calc: Calculator, element: str, bond_lengths: np.ndarray
) -> np.ndarray:
    """Evaluate model energy for a homonuclear dimer at each bond length (eV).

    Isolated dimer in a 30 Å cubic cell with PBC, matching the MLIP Arena setup.
    """
    cell = 30.0
    energies = []
    for r in bond_lengths:
        atoms = Atoms(
            symbols=[element, element],
            positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
            cell=[cell, cell, cell],
            pbc=True,
        )
        atoms.calc = calc
        try:
            e = atoms.get_potential_energy()
            if not np.isfinite(e):
                raise ValueError("non-finite energy")
        except Exception as exc:
            logging.warning(f"{element}2 @ r={r:.3f} Å failed: {exc}")
            e = np.nan
        energies.append(e)
    return np.array(energies)


def run_inference(model: ASEModel, test_data: Path | None = None) -> dict[str, dict]:
    """Evaluate curvature roughness on homonuclear dimers."""
    label_path = _LABEL_FILE if test_data is None else test_data / "diatomics.json"

    with open(label_path) as fh:
        reference_data: list[dict] = json.load(fh)

    results: dict[str, dict] = {}
    calc = model.calc

    for entry in reference_data:
        mol_name: str = entry["name"]
        bond_lengths, dft_energies = _scan_arrays(entry)
        element = _element_from_name(mol_name)
        dr = float(np.mean(np.diff(bond_lengths)))

        model_energies = _predict_energies(calc, element, bond_lengths)

        mol_result = {
            "roughness": _compute_roughness(model_energies - dft_energies, dr),
        }
        results[mol_name] = mol_result
        logging.info(f"{mol_name}: {mol_result}")

    return results
