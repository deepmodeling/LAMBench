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

Same comparison as stacking_fault, on the DFT bond lengths (no PCHIP):
shift each scan so min(E)=0, then MAE of d(E / max(E)) / dr, divided
by a constant-energy dummy (zeros after the shift) and capped at 1:

    0 = perfect match, 1 = dummy or worse.

    roughness – slope MAE relative to dummy; leaderboard

The last point of every scan is dropped (PBE tail artifacts).

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


def _shift_to_min(energies: np.ndarray) -> np.ndarray:
    return energies - np.min(energies)


def _normalized_slopes(bond_lengths: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d(E / max(E)) / dr on the native DFT scan (Å⁻¹)."""
    peak = float(np.max(y))
    slopes = np.diff(y) / np.diff(bond_lengths)
    if peak > 0:
        return slopes / peak
    return np.zeros_like(slopes)


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def _ratio_capped_at_dummy(value: float, dummy: float) -> float | None:
    if dummy <= 0 or not np.isfinite(dummy) or not np.isfinite(value):
        return None
    return float(min(value / dummy, 1.0))


def _curve_metrics(
    bond_lengths: np.ndarray, model_energies: np.ndarray, dft_energies: np.ndarray
) -> dict[str, float] | None:
    if not np.all(np.isfinite(model_energies)) or not np.all(np.isfinite(dft_energies)):
        return None
    if bond_lengths.size < 2:
        return None
    if (
        bond_lengths.size != model_energies.size
        or bond_lengths.size != dft_energies.size
    ):
        return None
    if np.any(np.diff(bond_lengths) == 0):
        return None

    y_dft = _shift_to_min(dft_energies)
    y_model = _shift_to_min(model_energies)
    y_dummy = np.zeros_like(y_dft)

    roughness = _ratio_capped_at_dummy(
        _mae(
            _normalized_slopes(bond_lengths, y_model),
            _normalized_slopes(bond_lengths, y_dft),
        ),
        _mae(
            _normalized_slopes(bond_lengths, y_dummy),
            _normalized_slopes(bond_lengths, y_dft),
        ),
    )
    if roughness is None:
        return None
    return {"roughness": roughness}


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
    """Compare model and PBE dissociation curves with the stacking_fault metric."""
    label_path = _LABEL_FILE if test_data is None else test_data / "diatomics.json"

    with open(label_path) as fh:
        reference_data: list[dict] = json.load(fh)

    results: dict[str, dict] = {}
    calc = model.calc

    for entry in reference_data:
        mol_name: str = entry["name"]
        bond_lengths, dft_energies = _scan_arrays(entry)
        element = _element_from_name(mol_name)
        model_energies = _predict_energies(calc, element, bond_lengths)
        mol_result = _curve_metrics(bond_lengths, model_energies, dft_energies)
        results[mol_name] = mol_result
        logging.info(f"{mol_name}: {mol_result}")

    return results
