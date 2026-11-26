"""Run ASE predictions for the long-range electrostatics benchmark."""

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read

from lambench.tasks.calculator.electrostatics.utils import (
    Prediction,
    Predictions,
    curve_errors,
    read_csv,
    restoring_forces,
    tail_fits,
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
    results = {
        "power_law": run_power_law(
            inputs / "twobody", reference, calc, dp_input, cache
        ),
        "charge_transfer": run_charge_transfer(
            inputs / "twobody", reference, calc, dp_input, cache
        ),
        "halogen_localization": run_halogen_localization(
            inputs / "hl", reference, calc, dp_input, cache
        ),
    }
    if not any(cache.values()):
        raise RuntimeError("No valid electrostatics predictions")
    return results


def run_power_law(data, reference, calc, dp_input, cache) -> dict:
    series = read_csv(reference / "tail_fits.csv")
    predictions = _predict_systems(data, series, calc, dp_input, cache)
    return {"tail_fits": tail_fits(reference, series, predictions)}


def run_charge_transfer(data, reference, calc, dp_input, cache) -> dict:
    rows = [
        row for row in read_csv(reference / "ct_points.csv") if row["benchmark"] == "ct"
    ]
    predictions = _predict_systems(data, rows, calc, dp_input, cache)
    return {"aligned_energy_errors": curve_errors(rows, predictions)}


def run_halogen_localization(data, reference, calc, dp_input, cache) -> dict:
    rows = [
        row
        for row in read_csv(reference / "hl.csv")
        if row["system"] == "HLA" and row["orientation"] in {"r=1.9", "r=2.2"}
    ]
    with (data / "hl_refs.json").open() as file:
        fragments = json.load(file)
    predictions = _predict_systems(data, rows, calc, dp_input, cache, fragments)
    return {
        "aligned_energy_errors": curve_errors(
            rows, predictions, split_orientations=True
        ),
        "restoring_forces": restoring_forces(rows, predictions),
    }


def _predict_systems(data, rows, calc, dp_input, cache, hl_refs=None) -> Predictions:
    paths = sorted(data.glob("*.xyz"))
    if not paths:
        raise FileNotFoundError(f"Expected XYZ files in {data}")

    systems = {row["system"] for row in rows}
    predictions: Predictions = {}
    for path in paths:
        if path.stem not in systems:
            continue
        if path not in cache:
            cache[path] = _predict_system(
                path, calc, dp_input, (hl_refs or {}).get(path.stem)
            )
        predictions.update(cache[path])
    return predictions


def _predict_system(path, calc, dp_input, hl) -> Predictions:
    predictions: Predictions = {}
    frames = read(path, index=":", format="extxyz")
    system = path.stem
    try:
        fragment_energy = _fragment_energy(frames[0], hl, calc, dp_input)
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

        force = None
        if hl is not None:
            try:
                # The HL bond lies along the x axis.
                first_atom = _selection(info["selection_mol"]).start
                force = float(atoms.get_forces()[first_atom, 0])
            except PREDICTION_ERRORS as exc:
                logging.warning("Electrostatics %s force failed: %s", frame_id, exc)

        predictions[system, frame_id] = Prediction(
            energy_kcal_mol=(energy - fragment_energy) * EV_TO_KCAL_MOL,
            force_x_ev_angstrom=force,
            distance=float(info["R"]),
            orientation=info["orient"],
            grid=str(info["grid"]),
        )
    return predictions


def _fragment_energy(
    first: Atoms, hl: dict | None, calc, dp_input: str | None
) -> float:
    info = first.info
    if hl is None:
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

    molecule = first[_selection(info["selection_mol"])]
    spectator = _selection(info["selection_spec"]).start
    molecule_energy = _energy(
        molecule, hl["mol_charge"], hl["mol_mult"], calc, dp_input
    )
    spectator_energy = _energy(
        first[spectator : spectator + 1],
        hl["spec_charge"],
        hl["spec_mult"],
        calc,
        dp_input,
    )
    return molecule_energy + hl["n_spec"] * spectator_energy


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
