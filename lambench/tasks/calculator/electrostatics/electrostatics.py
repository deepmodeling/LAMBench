"""Long-range electrostatics against frozen vacuum DFT reference curves."""

import csv
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read

TAIL_POWERS = {"QQ": 1, "QP": 2, "PP": 3}
CT_GROUPS = {
    "salts": ("CT01", "CT02", "CT03", "CT04", "CT05", "CT08"),
    "highq": ("QQ09", "QP09", "QP10", "QP16"),
    "hl_all": ("HLA",),
}
TAIL_MIN_DISTANCE = 10.0
TAIL_MIN_R2 = 0.95
HL_BONDS = (1.9, 2.2)
HL_MIN_FORCE = 0.05
MIN_RELATIVE_ENERGY = 0.1
EV_TO_KCAL_MOL = units.mol / units.kcal
PREDICTION_ERRORS = (ValueError, RuntimeError, KeyError, NotImplementedError)

EnergyEvaluator = Callable[[Atoms, int, int], float]
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


def _selection(value: str | int) -> slice:
    """Convert a one-based inclusive fragment selection to an ASE slice."""
    first, _, last = str(value).partition("-")
    return slice(int(first) - 1, int(last or first))


def _energy_evaluator(model) -> EnergyEvaluator:
    calc = model.calc
    dp_input = None
    if model.model_family == "DP":
        if getattr(model.model_metadata, "use_separate_spin_charge", False):
            dp_input = "charge_spin"
        elif calc.dp.get_dim_fparam() == 2:
            dp_input = "fparam"

    def evaluate(atoms, charge, spin):
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

    return evaluate


def _fragment_energy(first: Atoms, hl: dict | None, evaluate: EnergyEvaluator) -> float:
    info = first.info
    if hl is None:
        return sum(
            evaluate(
                first[_selection(info[f"selection_{part}"])],
                info[f"charge_{part}"],
                info.get("spin", 1),
            )
            for part in ("a", "b")
        )

    molecule = first[_selection(info["selection_mol"])]
    spectator = _selection(info["selection_spec"]).start
    molecule_energy = evaluate(molecule, hl["mol_charge"], hl["mol_mult"])
    spectator_energy = evaluate(
        first[spectator : spectator + 1], hl["spec_charge"], hl["spec_mult"]
    )
    return molecule_energy + hl["n_spec"] * spectator_energy


def _hl_force(atoms: Atoms, frame_id: str) -> float | None:
    try:
        # The HL bond lies along the x axis.
        first_atom = _selection(atoms.info["selection_mol"]).start
        return float(atoms.get_forces()[first_atom, 0])
    except PREDICTION_ERRORS as exc:
        logging.warning("Electrostatics %s force failed: %s", frame_id, exc)
        return None


def _predict(model, paths: list[Path], hl_refs: dict) -> Predictions:
    evaluate = _energy_evaluator(model)
    predictions = {}
    for path in paths:
        frames = read(path, index=":", format="extxyz")
        system = path.stem
        hl = hl_refs.get(system)
        try:
            fragment_energy = _fragment_energy(frames[0], hl, evaluate)
        except PREDICTION_ERRORS as exc:
            logging.warning(
                "Electrostatics %s fragment evaluation failed: %s", system, exc
            )
            continue

        for atoms in frames:
            info = atoms.info
            frame_id = str(info["frame"])
            try:
                energy = evaluate(atoms, info["charge"], info.get("spin", 1))
            except PREDICTION_ERRORS as exc:
                logging.warning("Electrostatics %s energy failed: %s", frame_id, exc)
                continue

            predictions[system, frame_id] = Prediction(
                energy_kcal_mol=(energy - fragment_energy) * EV_TO_KCAL_MOL,
                force_x_ev_angstrom=_hl_force(atoms, frame_id)
                if hl is not None
                else None,
                distance=float(info["R"]),
                orientation=info["orient"],
                grid=str(info["grid"]),
            )
    return predictions


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


def _tail_metrics(reference: Path, predictions: Predictions) -> dict:
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

    summary = {}
    fits = _read_csv(reference / "tail_fits.csv")
    for category, power in TAIL_POWERS.items():
        series = [row for row in fits if row["category"] == category]
        evaluated = passed = sign_errors = 0
        errors = []
        for row in series:
            key = row["system"], row["orientation"]
            radii = distances[key]
            values = points[key]
            if len(radii) < 3 or any(radius not in values for radius in radii):
                continue
            evaluated += 1
            model_a, r2 = _fit_power_law(
                radii, [values[radius] for radius in radii], power
            )
            if r2 is not None and r2 > TAIL_MIN_R2:
                passed += 1
            ref_a = float(row["A_kcal"])
            if np.sign(model_a) == np.sign(ref_a):
                errors.append(abs(1 - model_a / ref_a))
            else:
                sign_errors += 1
        summary[category] = {
            "n_series": len(series),
            "n_evaluated": evaluated,
            "n_pass_r2": passed,
            "sign_errors": sign_errors,
            "n_eps": len(errors),
            "median_abs_eps": float(np.median(errors)) if errors else None,
            "mean_abs_eps": float(np.mean(errors)) if errors else None,
            "max_abs_eps": max(errors) if errors else None,
        }
    return summary


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


def _restoring_force_metrics(
    rows: list[ReferenceRow], predictions: Predictions
) -> dict:
    ratios = []
    expected = 0
    for row in rows:
        ref = float(row["F_bond_dft_eVA"])
        if not (float(row["r_bond"]) in HL_BONDS and abs(ref) > HL_MIN_FORCE):
            continue
        expected += 1
        prediction = predictions.get((row["system"], row["frame_id"]))
        force = prediction.force_x_ev_angstrom if prediction is not None else None
        if force is not None and np.isfinite(force):
            ratios.append(force / ref)

    return {
        "n_expected": expected,
        "n_evaluated": len(ratios),
        "n_correct_sign": sum(ratio > 0 for ratio in ratios),
        "median_ratio": float(np.median(ratios)) if ratios else None,
    }


def _curve_metrics(reference: Path, predictions: Predictions) -> dict:
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

    maes = {system: float(np.mean(values)) for system, values in errors.items()}
    relative = [
        maes[system] / float(np.mean(changes[system]))
        for system in maes
        if np.mean(changes[system]) > MIN_RELATIVE_ENERGY
    ]

    groups = {}
    for group, systems in CT_GROUPS.items():
        values = [maes[system] for system in systems if system in maes]
        groups[group] = {
            "mae_kcal_mol": float(np.mean(values)) if values else None,
            "n_evaluated": len(values),
            "n_systems": len(systems),
        }

    return {
        "groups": groups,
        "median_mae_kcal_mol": float(np.median(list(maes.values()))) if maes else None,
        "median_relative_mae": float(np.median(relative)) if relative else None,
        "restoring_force": _restoring_force_metrics(hl_rows, predictions),
    }


def run_inference(model, data: Path) -> dict:
    inputs = data / "inputs"
    reference = data / "reference"
    twobody = sorted((inputs / "twobody").glob("*.xyz"))
    hl = sorted((inputs / "hl").glob("*.xyz"))
    if not twobody or not hl:
        raise FileNotFoundError(
            f"Expected inputs/twobody and inputs/hl XYZ files in {data}"
        )
    with (inputs / "hl" / "hl_refs.json").open() as file:
        hl_refs = json.load(file)
    predictions = _predict(model, twobody + hl, hl_refs)
    if not predictions:
        raise RuntimeError("No valid electrostatics predictions")
    return {
        "power_law": _tail_metrics(reference, predictions),
        "charge_transfer_and_halogen_localization": _curve_metrics(
            reference, predictions
        ),
    }
