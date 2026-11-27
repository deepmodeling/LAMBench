"""Score electrostatics predictions against the frozen reference series."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAIL_POWERS = {"QQ": 1, "QP": 2, "PP": 3}
TAIL_MIN_DISTANCE = 10.0
MIN_RELATIVE_COEFFICIENT = 1e-3

ReferenceRow = dict[str, str]


@dataclass(frozen=True)
class Prediction:
    energy_kcal_mol: float
    distance: float
    orientation: str
    grid: str


Predictions = dict[tuple[str, str], Prediction]


def read_csv(path: Path) -> list[ReferenceRow]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _fit_power_law(distances: list[float], energies: list[float], power: int) -> float:
    """Fit E = A / R**power + B and return A."""
    if not np.ptp(energies):
        return 0.0

    design = np.column_stack([np.asarray(distances) ** -power, np.ones(len(distances))])
    coefficients = np.linalg.lstsq(design, energies, rcond=None)[0]
    return float(coefficients[0])


def power_law_mae(
    reference: Path, series: list[ReferenceRow], predictions: Predictions
) -> dict:
    points = defaultdict(dict)
    for (system, _), prediction in predictions.items():
        if (
            prediction.grid.startswith("radial")
            and prediction.distance > TAIL_MIN_DISTANCE
        ):
            series_points = points[system, prediction.orientation]
            series_points[prediction.distance] = prediction.energy_kcal_mol

    distances = defaultdict(list)
    for row in read_csv(reference / "tail_points.csv"):
        distances[row["system"], row["orientation"]].append(float(row["R"]))

    errors = defaultdict(list)
    responsive_categories = set()
    for row in series:
        category = row["category"]
        key = row["system"], row["orientation"]
        radii = distances[key]
        values = points[key]
        if len(radii) < 3 or any(radius not in values for radius in radii):
            continue

        coefficient = _fit_power_law(
            radii, [values[radius] for radius in radii], TAIL_POWERS[category]
        )
        reference_coefficient = float(row["A_kcal"])
        errors[category].append(abs(coefficient - reference_coefficient))
        if abs(coefficient) > MIN_RELATIVE_COEFFICIENT * abs(reference_coefficient):
            responsive_categories.add(category)

    return {
        category: {
            "coefficient_mae": float(np.mean(errors[category]))
            if category in responsive_categories
            else None,
        }
        for category in TAIL_POWERS
    }


def charge_transfer_mae(rows: list[ReferenceRow], predictions: Predictions) -> dict:
    curves = defaultdict(list)
    for row in rows:
        curves[row["system"]].append(row)

    maes = []
    for system, curve in curves.items():
        ordered = sorted(curve, key=lambda row: float(row["R"]))
        keys = [(system, row["frame_id"]) for row in ordered]
        if len(keys) < 2 or any(key not in predictions for key in keys):
            continue

        model = np.array([predictions[key].energy_kcal_mol for key in keys])
        reference = np.array([float(row["E_int_kcal"]) for row in ordered])
        error = (model - model[-1]) - (reference - reference[-1])
        maes.append(float(np.mean(np.abs(error))))

    return {
        "mae_kcal_mol": float(np.mean(maes)) if maes else None,
    }
