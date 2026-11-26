"""Run ASE predictions for the long-range electrostatics benchmark."""

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read

from lambench.tasks.calculator.electrostatics.metrics import (
    Prediction,
    Predictions,
    curve_errors,
    tail_fits,
)

EV_TO_KCAL_MOL = units.mol / units.kcal
PREDICTION_ERRORS = (ValueError, RuntimeError, KeyError, NotImplementedError)


def run_inference(model, data: Path) -> dict:
    inputs = data / "inputs"
    reference = data / "reference"
    twobody = sorted((inputs / "twobody").glob("*.xyz"))
    hl_paths = sorted((inputs / "hl").glob("*.xyz"))
    if not twobody or not hl_paths:
        raise FileNotFoundError(
            f"Expected inputs/twobody and inputs/hl XYZ files in {data}"
        )
    with (inputs / "hl" / "hl_refs.json").open() as file:
        hl_refs = json.load(file)

    calc = model.calc
    dp_input = None
    if model.model_family == "DP":
        if getattr(model.model_metadata, "use_separate_spin_charge", False):
            dp_input = "charge_spin"
        elif calc.dp.get_dim_fparam() == 2:
            dp_input = "fparam"

    predictions: Predictions = {}
    for path in twobody + hl_paths:
        frames = read(path, index=":", format="extxyz")
        system = path.stem
        hl = hl_refs.get(system)
        try:
            fragment_energy = _fragment_energy(frames[0], hl, calc, dp_input)
        except PREDICTION_ERRORS as exc:
            logging.warning(
                "Electrostatics %s fragment evaluation failed: %s", system, exc
            )
            continue

        for atoms in frames:
            info = atoms.info
            frame_id = str(info["frame"])
            try:
                energy = _energy(
                    atoms, info["charge"], info.get("spin", 1), calc, dp_input
                )
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

    if not predictions:
        raise RuntimeError("No valid electrostatics predictions")
    return {
        "tail_fits": tail_fits(reference, predictions),
        **curve_errors(reference, predictions),
    }


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
