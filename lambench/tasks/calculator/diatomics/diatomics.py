"""Homonuclear diatomic curves (Applicability).

Data source
-----------
PBE curves for homonuclear dimers, from the Matbench Discovery diatomic
reference. The published file also contains r2SCAN; this task does not
score it.

    Matbench Discovery diatomic DFT curves (PBE and r2SCAN, H–U)
    https://figshare.com/files/68541277

    Provenance, VASP settings, and spin ladder
    https://github.com/janosh/matbench-discovery/blob/main/site/src/lib/diatomics-dft.readme.md

    Riebesell, J., Goodall, R. E. A., Benner, P. et al. A framework to
    evaluate machine learning crystal stability predictions.
    Nat. Mach. Intell. 7, 836–847 (2025).
    https://doi.org/10.1038/s42256-025-01055-1

Each curve was computed with VASP 6, PBE_64 PAW potentials, and Materials
Project MP24 static settings in a 15 Å cell. Distances run geometrically
from 0.8 covalent radii to 6 Å. At each distance the stored energy is the
lowest among an even-NUPDOWN spin ladder plus one antiferromagnetic
candidate. The public file has no raw spin-candidate curves and no
per-point edit log, so a deleted distance cannot be restored. Po, At, Rn,
Fr, and Ra are not part of the scored set, and Ni, Zr, Pr, Pm, Sm, Tb, Dy,
Ho, Er, Tm, and Ir are missing 45 design-grid distances. Those curves are
already absent from ``diatomics.json``. The loader reads that file as
stored.

Three per-element metrics are averaged over finite values:

- bond_length_error: absolute equilibrium-distance error in Å. The distance
  is a quadratic fit of up to five points around the minimum. Reference
  binding energies below 0.05 eV are skipped.
- well_depth_error: absolute error in eV of
  ``E(r_far) - min(E)`` over the evaluation window. It is skipped with
  bond length when the reference binding energy is below 0.05 eV.
- force_flip_count: number of Fx sign changes on the first atom after
  dropping |Fx| < 0.01 eV/Å.

All metrics use distances from 0.9 covalent radii to
min(3.1 Alvarez vdW radii, 6 Å). A model curve that is non-finite in that
range is dropped from the averages and lowers coverage.

The leaderboard score is the equal-weight average of the dummy-normalized
bond-length MAE, dummy-normalized well-depth MAE, and mean absolute
deviation from one force flip, divided by coverage. The bond-length dummy
guesses the midpoint of each element's evaluation window. The well-depth
dummy predicts zero binding energy. Zero is a perfect single-well match.

Reference file: lambench/tasks/calculator/diatomics/diatomics.json
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers, covalent_radii, vdw_alvarez

if TYPE_CHECKING:
    from lambench.models.ase_models import ASEModel

_LABEL_FILE = Path(__file__).parent / "diatomics.json"
_N_GRID = 50
_GRID_R_MAX = 6.0
_MIN_BINDING_EV = 0.05
_MIN_WINDOW_POINTS = 5
_FORCE_SIGN_THRESHOLD = 1e-2
_FIT_POINTS = 5
_BOX_ANGSTROM = 30.0


def design_distances(element: str) -> np.ndarray:
    """50 geometric separations from 0.8 covalent radii to 6 Å."""
    r_min = 0.8 * float(covalent_radii[atomic_numbers[element]])
    return r_min * (_GRID_R_MAX / r_min) ** (np.arange(_N_GRID) / (_N_GRID - 1))


def eval_window(
    element: str, seps_max: float, *, r_min_factor: float = 0.9
) -> tuple[float, float]:
    """Element-specific scoring window in Å.

    The default lower bound is 0.9 covalent radii. Wall scoring passes
    ``r_min_factor=0.8``. The upper bound is min(3.1 Alvarez vdW radii,
    ``seps_max``). Missing vdW radii leave the upper bound at ``seps_max``.
    """
    atomic_num = atomic_numbers[element]
    r_cov = float(covalent_radii[atomic_num])
    r_min = r_min_factor * r_cov if np.isfinite(r_cov) else 0.0
    r_vdw = float(vdw_alvarez.vdw_radii[atomic_num])
    r_max = min(3.1 * r_vdw, seps_max) if np.isfinite(r_vdw) else float(seps_max)
    return r_min, r_max


def _binding_energy(energies: np.ndarray) -> float:
    return float(energies[-1] - np.min(energies))


def _equilibrium_distance(seps: np.ndarray, energies: np.ndarray) -> float:
    """Equilibrium separation from a local quadratic fit, in Å."""
    min_idx = int(np.argmin(energies))
    if len(seps) < 3:
        return float(seps[min_idx])
    start_idx = min(max(0, min_idx - _FIT_POINTS // 2), max(0, len(seps) - _FIT_POINTS))
    fit_seps = seps[start_idx : start_idx + _FIT_POINTS]
    fit_energies = energies[start_idx : start_idx + _FIT_POINTS]
    if len(fit_seps) < 3:
        return float(seps[min_idx])
    quadratic_coef, linear_coef, _constant = np.polyfit(fit_seps, fit_energies, 2)
    if quadratic_coef <= 0:
        return float(seps[min_idx])
    equilibrium = -linear_coef / (2 * quadratic_coef)
    if fit_seps.min() <= equilibrium <= fit_seps.max():
        return float(equilibrium)
    return float(seps[min_idx])


def _force_flip_count(force_x: np.ndarray) -> int:
    kept = force_x[np.abs(force_x) >= _FORCE_SIGN_THRESHOLD]
    if kept.size < 2:
        return 0
    return int(np.sum(np.diff(np.sign(kept)) != 0))


def _excluded_record() -> dict[str, float | int | bool | None]:
    return {
        "bond_length_error": None,
        "well_depth_error": None,
        "force_flip_count": None,
        "model_excluded": True,
    }


def score_curve(
    element: str,
    distances: np.ndarray,
    ref_energies: np.ndarray,
    model_energies: np.ndarray,
    model_force_x: np.ndarray,
) -> dict[str, float | int | bool | None]:
    """Score one homonuclear curve against its PBE reference.

    ``model_excluded`` is true when the model curve cannot be scored.
    """
    distances = np.asarray(distances, dtype=float)
    ref_energies = np.asarray(ref_energies, dtype=float)
    model_energies = np.asarray(model_energies, dtype=float)
    model_force_x = np.asarray(model_force_x, dtype=float)
    if not (
        distances.size == ref_energies.size == model_energies.size == model_force_x.size
    ):
        return _excluded_record()
    if distances.size < 2 or np.any(np.diff(distances) <= 0):
        return _excluded_record()

    seps_max = float(distances[-1])
    r_min, r_max = eval_window(element, seps_max)
    general = (distances >= r_min) & (distances <= r_max)
    if int(general.sum()) < _MIN_WINDOW_POINTS:
        return _excluded_record()
    if not (
        np.isfinite(model_energies[general]).all()
        and np.isfinite(model_force_x[general]).all()
    ):
        return _excluded_record()
    if not np.isfinite(ref_energies[general]).all():
        return _excluded_record()

    n_flips = _force_flip_count(model_force_x[general])
    result: dict[str, float | int | bool | None] = {
        "bond_length_error": None,
        "well_depth_error": None,
        "force_flip_count": n_flips,
        "model_excluded": False,
    }

    ref_general = ref_energies[general]
    model_general = model_energies[general]
    ref_depth = _binding_energy(ref_general)
    if ref_depth >= _MIN_BINDING_EV:
        ref_distance = _equilibrium_distance(distances[general], ref_general)
        model_distance = _equilibrium_distance(distances[general], model_general)
        result["bond_length_error"] = float(abs(model_distance - ref_distance))
        result["well_depth_error"] = abs(_binding_energy(model_general) - ref_depth)
    return result


@lru_cache
def load_reference(
    path: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Scored PBE curves keyed by element, distances then energies in Å and eV."""
    label_path = _LABEL_FILE if path is None else Path(path)
    with open(label_path) as fh:
        raw: list[dict] = json.load(fh)
    return {
        entry["element"]: (
            np.asarray(entry["R"], dtype=float),
            np.asarray(entry["E"], dtype=float),
        )
        for entry in raw
    }


def scored_element_names(path: str | None = None) -> tuple[str, ...]:
    return tuple(load_reference(path))


@lru_cache
def reference_dummy_scales(path: str | None = None) -> dict[str, float]:
    """Metric scales from blind bond-position and flat-energy predictions.

    For each bound reference, the bond dummy guesses the arithmetic midpoint
    of that element's evaluation window and the well-depth dummy predicts zero.
    A flat force curve has zero flips, one away from the ideal count of one.
    """
    bond_errors: list[float] = []
    well_errors: list[float] = []
    for element, (distances, energies) in load_reference(path).items():
        seps_max = float(np.max(distances))
        r_min, r_max = eval_window(element, seps_max)
        general = (distances >= r_min) & (distances <= r_max)
        if int(general.sum()) < 3:
            continue
        ref_energies = energies[general]
        ref_depth = _binding_energy(ref_energies)
        if ref_depth < _MIN_BINDING_EV:
            continue
        ref_distance = _equilibrium_distance(distances[general], ref_energies)
        bond_errors.append(abs((r_min + r_max) / 2 - ref_distance))
        well_errors.append(ref_depth)

    return {
        "bond_length_mae": float(np.mean(bond_errors)),
        "well_depth_mae": float(np.mean(well_errors)),
        "force_flip_deviation": 1.0,
    }


def _predict_curve(
    calc: Calculator, element: str, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Model energy (eV) and Fx on the first atom (eV/Å) at each distance."""
    energies = np.empty(distances.size, dtype=float)
    force_x = np.empty(distances.size, dtype=float)
    for i, distance in enumerate(distances):
        atoms = Atoms(
            symbols=[element, element],
            positions=[[0.0, 0.0, 0.0], [float(distance), 0.0, 0.0]],
            cell=[_BOX_ANGSTROM, _BOX_ANGSTROM, _BOX_ANGSTROM],
            pbc=True,
        )
        atoms.calc = calc
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            fx = float(forces[0, 0])
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError("non-finite energy or forces")
        except Exception as exc:
            logging.warning(f"{element}2 @ r={float(distance):.3f} Å failed: {exc}")
            energy = np.nan
            fx = np.nan
        energies[i] = energy
        force_x[i] = fx
    return energies, force_x


def run_inference(model: ASEModel, test_data: Path | None = None) -> dict[str, dict]:
    """Score model curves on the PBE homonuclear reference grid."""
    label_path = None if test_data is None else str(test_data / "diatomics.json")
    reference = load_reference(label_path)
    calc = model.calc
    results: dict[str, dict] = {}
    for element, (distances, ref_energies) in reference.items():
        model_energies, model_force_x = _predict_curve(calc, element, distances)
        element_result = score_curve(
            element,
            distances,
            ref_energies,
            model_energies,
            model_force_x,
        )
        results[element] = element_result
        logging.info(f"{element}: {element_result}")
    return results
