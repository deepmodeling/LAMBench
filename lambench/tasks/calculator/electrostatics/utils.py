"""Score electrostatics predictions against the frozen reference series."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAIL_POWERS = {"QQ": 1, "QP": 2, "PP": 3}
TAIL_MIN_DISTANCE = 10.0
HL_BONDS = (1.9, 2.2)
HL_MIN_FORCE = 0.05

ReferenceRow = dict[str, str]


@dataclass(frozen=True)
class Prediction:
    energy_kcal_mol: float
    force_x_ev_angstrom: float | None
    distance: float
    orientation: str
    grid: str


Predictions = dict[tuple[str, str], Prediction]


def _read_csv(path: Path) -> list[ReferenceRow]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _fit_power_law(
    distances: list[float], energies: list[float], power: int
) -> tuple[float, float | None]:
    """Fit E = A / R**power + B and return A and R²."""
    energy = np.asarray(energies)
    if not np.ptp(energy):
        return 0.0, None

    design = np.column_stack([np.asarray(distances) ** -power, np.ones(len(distances))])
    coefficients = np.linalg.lstsq(design, energy, rcond=None)[0]
    variance = np.sum((energy - energy.mean()) ** 2)
    residual = np.sum((energy - design @ coefficients) ** 2)
    return float(coefficients[0]), float(1 - residual / variance)


def tail_fits(reference: Path, predictions: Predictions) -> dict:
    points = defaultdict(dict)
    for (system, _), prediction in predictions.items():
        if (
            prediction.grid.startswith("radial")
            and prediction.distance > TAIL_MIN_DISTANCE
        ):
            series = points[system, prediction.orientation]
            series[prediction.distance] = prediction.energy_kcal_mol

    distances = defaultdict(list)
    for row in _read_csv(reference / "tail_points.csv"):
        distances[row["system"], row["orientation"]].append(float(row["R"]))

    fits = {category: [] for category in TAIL_POWERS}
    for row in _read_csv(reference / "tail_fits.csv"):
        category = row["category"]
        key = row["system"], row["orientation"]
        radii = distances[key]
        values = points[key]
        coefficient = r2 = None
        if len(radii) >= 3 and all(radius in values for radius in radii):
            coefficient, r2 = _fit_power_law(
                radii, [values[radius] for radius in radii], TAIL_POWERS[category]
            )
        fits[category].append(
            {
                "system": row["system"],
                "orientation": row["orientation"],
                "coefficient": coefficient,
                "reference_coefficient": float(row["A_kcal"]),
                "r2": r2,
            }
        )
    return fits


def _aligned_curve_errors(
    rows: list[ReferenceRow], predictions: Predictions
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compare energy changes relative to the farthest reference point."""
    ordered = sorted(rows, key=lambda row: float(row["R"]))
    keys = [(row["system"], row["frame_id"]) for row in ordered]
    if len(keys) < 2 or any(key not in predictions for key in keys):
        return None

    model_energy = np.array([predictions[key].energy_kcal_mol for key in keys])
    ref_energy = np.array([float(row["E_int_kcal"]) for row in ordered])
    model_shift = model_energy - model_energy[-1]
    ref_shift = ref_energy - ref_energy[-1]
    return abs(model_shift - ref_shift), abs(ref_shift)


def curve_errors(reference: Path, predictions: Predictions) -> dict:
    ct_rows = [
        row
        for row in _read_csv(reference / "ct_points.csv")
        if row["benchmark"] == "ct"
    ]
    hl_rows = [
        row
        for row in _read_csv(reference / "hl.csv")
        if row["system"] == "HLA" and row["orientation"] in {"r=1.9", "r=2.2"}
    ]
    curves = defaultdict(list)
    for row in ct_rows + hl_rows:
        branch = row["orientation"] if row["system"] == "HLA" else ""
        curves[row["system"], branch].append(row)

    errors = defaultdict(list)
    changes = defaultdict(list)
    for (system, _), rows in curves.items():
        aligned = _aligned_curve_errors(rows, predictions)
        if aligned is None:
            continue
        error, change = aligned
        errors[system].extend(error)
        changes[system].extend(change)

    # Keep HLA comparable by requiring both frozen branches.
    if any((row["system"], row["frame_id"]) not in predictions for row in hl_rows):
        errors.pop("HLA", None)

    forces = {}
    for row in hl_rows:
        ref = float(row["F_bond_dft_eVA"])
        if not (float(row["r_bond"]) in HL_BONDS and abs(ref) > HL_MIN_FORCE):
            continue
        prediction = predictions.get((row["system"], row["frame_id"]))
        force = prediction.force_x_ev_angstrom if prediction is not None else None
        forces[row["frame_id"]] = {
            "predicted_ev_angstrom": force
            if force is not None and np.isfinite(force)
            else None,
            "reference_ev_angstrom": ref,
        }

    return {
        "aligned_energy_errors": {
            system: {
                "mae_kcal_mol": float(np.mean(values)),
                "reference_change_kcal_mol": float(np.mean(changes[system])),
            }
            for system, values in errors.items()
        },
        "restoring_forces": forces,
    }
