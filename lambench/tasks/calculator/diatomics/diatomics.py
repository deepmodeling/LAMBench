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
  binding energies below 0.05 eV are skipped, as are PBE curves that fail
  the smoothness gate.
- wall_dist_error: MAE in Å of repulsive-branch radii at 1, 5, 10, 20, 50,
  and 100 eV above the well. A threshold the reference reaches and the
  model does not contributes the full reference radius.
- force_flip_fail: 0 when Fx on the first atom changes sign exactly once
  after dropping |Fx| < 0.01 eV/Å, otherwise 1.

Geometry metrics use distances from 0.9 covalent radii to
min(3.1 Alvarez vdW radii, 6 Å). The wall metric extends the lower bound
to 0.8 covalent radii. A model curve that is non-finite in that range, or
that has an energy jump of at least 1.5 eV and at least three
energy-difference sign flips, is dropped from the averages and lowers
coverage.

The leaderboard score is the equal-weight average of the dummy-normalized
bond-length MAE, the dummy-normalized wall MAE, and the force-flip rate,
divided by coverage. The dummy predicts each geometric quantity by the
mean of the PBE reference. Zero is a perfect single-well match.

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
_WALL_THRESHOLDS_EV = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
_MIN_BINDING_EV = 0.05
_MIN_ENERGY_JUMP_EV = 1.5
_MIN_ENERGY_FLIPS = 3
_MIN_WINDOW_POINTS = 5
_FORCE_SIGN_THRESHOLD = 1e-2
_ENERGY_DIFF_THRESHOLD = 1e-3
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


def _energy_smoothness(energies: np.ndarray) -> tuple[float, int]:
    """Energy jump (eV) at sign flips, and the flip count.

    Differences smaller than 1e-3 eV are treated as zero, matching the
    Matbench Discovery gate.
    """
    diffs = np.diff(np.asarray(energies, dtype=float))
    diffs[np.abs(diffs) < _ENERGY_DIFF_THRESHOLD] = 0
    signs = np.sign(diffs)
    kept = signs != 0
    diffs = diffs[kept]
    signs = signs[kept]
    if signs.size < 2:
        return 0.0, 0
    flips = np.diff(signs) != 0
    jump = float(np.abs(diffs[:-1][flips]).sum() + np.abs(diffs[1:][flips]).sum())
    return jump, int(np.sum(flips))


def _fails_smoothness_gate(energies: np.ndarray) -> bool:
    jump, n_flips = _energy_smoothness(energies)
    return jump >= _MIN_ENERGY_JUMP_EV and n_flips >= _MIN_ENERGY_FLIPS


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


def _repulsive_radius(
    seps: np.ndarray, energies: np.ndarray, threshold_ev: float
) -> float:
    """Invert the repulsive branch to the radius at E_min + threshold_ev."""
    min_idx = int(np.argmin(energies))
    if min_idx == 0:
        return np.nan
    radii_inward = seps[min_idx::-1]
    energy_above_min = energies[min_idx::-1] - energies[min_idx]
    monotonic_energy = np.maximum.accumulate(energy_above_min)
    unique_energy, unique_idx = np.unique(monotonic_energy, return_index=True)
    if len(unique_energy) < 2 or threshold_ev > unique_energy[-1]:
        return np.nan
    return float(np.interp(threshold_ev, unique_energy, radii_inward[unique_idx]))


def _wall_distance_mae(
    ref_seps: np.ndarray,
    ref_energies: np.ndarray,
    model_seps: np.ndarray,
    model_energies: np.ndarray,
) -> float | None:
    errors: list[float] = []
    for threshold in _WALL_THRESHOLDS_EV:
        radius_ref = _repulsive_radius(ref_seps, ref_energies, threshold)
        if not np.isfinite(radius_ref):
            continue
        radius_pred = _repulsive_radius(model_seps, model_energies, threshold)
        errors.append(
            abs(radius_pred - radius_ref) if np.isfinite(radius_pred) else radius_ref
        )
    if not errors:
        return None
    return float(np.mean(errors))


def _force_flip_count(force_x: np.ndarray) -> int:
    kept = force_x[np.abs(force_x) >= _FORCE_SIGN_THRESHOLD]
    if kept.size < 2:
        return 0
    return int(np.sum(np.diff(np.sign(kept)) != 0))


def _excluded_record() -> dict[str, float | int | bool | None]:
    return {
        "bond_length_error": None,
        "wall_dist_error": None,
        "force_flip_fail": None,
        "model_excluded": True,
    }


def score_curve(
    element: str,
    distances: np.ndarray,
    ref_energies: np.ndarray,
    model_energies: np.ndarray,
    model_force_x: np.ndarray,
    *,
    reference_low_quality: bool,
) -> dict[str, float | int | bool | None]:
    """Score one homonuclear curve against its PBE reference.

    ``model_excluded`` is true when the model curve cannot be scored.
    Reference-quality skips leave ``model_excluded`` false and set the
    PBE-relative errors to None.
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
    wall_r_min = eval_window(element, seps_max, r_min_factor=0.8)[0] - 1e-12
    general = (distances >= r_min) & (distances <= r_max)
    wall = (distances >= wall_r_min) & (distances <= r_max)
    if int(general.sum()) < _MIN_WINDOW_POINTS or int(wall.sum()) < 2:
        return _excluded_record()
    if not (
        np.isfinite(model_energies[wall]).all()
        and np.isfinite(model_force_x[wall]).all()
    ):
        return _excluded_record()
    if not np.isfinite(ref_energies[wall]).all():
        return _excluded_record()
    if _fails_smoothness_gate(model_energies[general]):
        return _excluded_record()

    n_flips = _force_flip_count(model_force_x[general])
    result: dict[str, float | int | bool | None] = {
        "bond_length_error": None,
        "wall_dist_error": None,
        "force_flip_fail": 0 if n_flips == 1 else 1,
        "model_excluded": False,
    }
    if reference_low_quality:
        return result

    result["wall_dist_error"] = _wall_distance_mae(
        distances[wall],
        ref_energies[wall],
        distances[wall],
        model_energies[wall],
    )
    ref_general = ref_energies[general]
    if _binding_energy(ref_general) >= _MIN_BINDING_EV:
        ref_distance = _equilibrium_distance(distances[general], ref_general)
        model_distance = _equilibrium_distance(
            distances[general], model_energies[general]
        )
        result["bond_length_error"] = float(abs(model_distance - ref_distance))
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
def low_quality_elements(path: str | None = None) -> frozenset[str]:
    """PBE curves too jumpy to use as a geometry reference."""
    flagged: set[str] = set()
    for element, (distances, energies) in load_reference(path).items():
        r_min, r_max = eval_window(element, float(np.max(distances)))
        mask = (distances >= r_min) & (distances <= r_max)
        if int(mask.sum()) < _MIN_WINDOW_POINTS:
            continue
        window = energies[mask]
        if not np.isfinite(window).all() or _fails_smoothness_gate(window):
            flagged.add(element)
    return frozenset(flagged)


@lru_cache
def reference_dummy_scales(path: str | None = None) -> dict[str, float]:
    """MAE of predicting each reference geometry by its cross-element mean.

    Bond lengths use elements that pass the smoothness gate and bind by at
    least 0.05 eV. Wall radii use every gated element that reaches a
    threshold; the dummy error at each threshold is the deviation from the
    mean radius at that threshold.
    """
    low_quality = low_quality_elements(path)
    bond_lengths: list[float] = []
    radii: dict[float, dict[str, float]] = {t: {} for t in _WALL_THRESHOLDS_EV}
    for element, (distances, energies) in load_reference(path).items():
        if element in low_quality:
            continue
        seps_max = float(np.max(distances))
        r_min, r_max = eval_window(element, seps_max)
        general = (distances >= r_min) & (distances <= r_max)
        if (
            int(general.sum()) >= 3
            and _binding_energy(energies[general]) >= _MIN_BINDING_EV
        ):
            bond_lengths.append(
                _equilibrium_distance(distances[general], energies[general])
            )
        wall_r_min = eval_window(element, seps_max, r_min_factor=0.8)[0] - 1e-12
        wall = (distances >= wall_r_min) & (distances <= r_max)
        if int(wall.sum()) < 2:
            continue
        for threshold in _WALL_THRESHOLDS_EV:
            radius = _repulsive_radius(distances[wall], energies[wall], threshold)
            if np.isfinite(radius):
                radii[threshold][element] = radius

    bond = np.asarray(bond_lengths, dtype=float)
    bond_dummy = float(np.mean(np.abs(bond - np.mean(bond))))
    per_element: dict[str, list[float]] = {}
    for threshold, values in radii.items():
        if not values:
            continue
        mean_radius = float(np.mean(list(values.values())))
        for element, radius in values.items():
            per_element.setdefault(element, []).append(abs(radius - mean_radius))
    wall_errors = [float(np.mean(errs)) for errs in per_element.values() if errs]
    return {
        "bond_length_mae": bond_dummy,
        "wall_dist_mae": float(np.mean(wall_errors)),
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
    low_quality = low_quality_elements(label_path)
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
            reference_low_quality=element in low_quality,
        )
        results[element] = element_result
        logging.info(f"{element}: {element_result}")
    return results
