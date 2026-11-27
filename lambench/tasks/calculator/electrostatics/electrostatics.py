"""Run ASE predictions for the long-range electrostatics benchmark."""

import logging
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read

from lambench.tasks.calculator.electrostatics.utils import (
    Prediction,
    Predictions,
    charge_transfer_mae,
    read_csv,
    power_law_mae,
)

EV_TO_KCAL_MOL = units.mol / units.kcal
PREDICTION_ERRORS = (ValueError, RuntimeError, KeyError, NotImplementedError)


def run_inference(model, data: Path) -> dict:
    inputs = data / "inputs"
    reference = data / "reference"
    calc = model.calc
    dp_input = None
    if model.model_family == "DP":
        if getattr(model.model_metadata, "use_separate_spin_charge", False):
            dp_input = "charge_spin"
        elif calc.dp.get_dim_fparam() == 2:
            dp_input = "fparam"

    cache: dict[Path, Predictions] = {}
    power_law = run_power_law(inputs / "twobody", reference, calc, dp_input, cache)
    charge_transfer = None
    if any(
        category["coefficient_relative_mae"] is not None
        for category in power_law.values()
    ):
        charge_transfer = run_charge_transfer(
            inputs / "twobody", reference, calc, dp_input, cache
        )
    return {"power_law": power_law, "charge_transfer": charge_transfer}


def run_power_law(data, reference, calc, dp_input, cache) -> dict:
    series = read_csv(reference / "tail_fits.csv")
    predictions = _predict_systems(data, series, calc, dp_input, cache)
    return power_law_mae(reference, series, predictions)


def run_charge_transfer(data, reference, calc, dp_input, cache) -> dict:
    rows = [
        row for row in read_csv(reference / "ct_points.csv") if row["benchmark"] == "ct"
    ]
    predictions = _predict_systems(data, rows, calc, dp_input, cache)
    return charge_transfer_mae(rows, predictions)


def _predict_systems(data, rows, calc, dp_input, cache) -> Predictions:
    paths = sorted(data.glob("*.xyz"))
    if not paths:
        raise FileNotFoundError(f"Expected XYZ files in {data}")

    systems = {row["system"] for row in rows}
    predictions: Predictions = {}
    for path in paths:
        if path.stem not in systems:
            continue
        if path not in cache:
            cache[path] = _predict_system(path, calc, dp_input)
        predictions.update(cache[path])
    return predictions


def _predict_system(path, calc, dp_input) -> Predictions:
    predictions: Predictions = {}
    frames = read(path, index=":", format="extxyz")
    system = path.stem
    try:
        fragment_energy = _fragment_energy(frames[0], calc, dp_input)
    except PREDICTION_ERRORS as exc:
        logging.warning("Electrostatics %s fragment evaluation failed: %s", system, exc)
        return predictions

    for atoms in frames:
        info = atoms.info
        frame_id = str(info["frame"])
        try:
            energy = _energy(atoms, info["charge"], info.get("spin", 1), calc, dp_input)
        except PREDICTION_ERRORS as exc:
            logging.warning("Electrostatics %s energy failed: %s", frame_id, exc)
            continue

        predictions[system, frame_id] = Prediction(
            energy_kcal_mol=(energy - fragment_energy) * EV_TO_KCAL_MOL,
            distance=float(info["R"]),
            orientation=info["orient"],
            grid=str(info["grid"]),
        )
    return predictions


def _fragment_energy(first: Atoms, calc, dp_input: str | None) -> float:
    info = first.info
    return sum(
        _energy(
            first[_selection(info[f"selection_{part}"])],
            info[f"charge_{part}"],
            info.get("spin", 1),
            calc,
            dp_input,
        )
        for part in ("a", "b")
    )


def _energy(atoms: Atoms, charge: int, spin: int, calc, dp_input: str | None) -> float:
    atoms.info.update(charge=int(charge), spin=int(spin))
    if dp_input is not None:
        atoms.info[dp_input] = np.array([[charge, spin]], dtype=float)
    # ASE's default cache check does not consider atoms.info.
    calc.reset()
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(energy):
        raise ValueError("Non-finite energy")
    return energy


def _selection(value: str | int) -> slice:
    """Convert a one-based inclusive fragment selection to an ASE slice."""
    first, _, last = str(value).partition("-")
    return slice(int(first) - 1, int(last or first))
