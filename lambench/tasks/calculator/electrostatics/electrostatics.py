"""Long-range electrostatics against frozen vacuum DFT reference curves."""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read

TAIL_POWERS = {"QQ": 1, "QP": 2, "PP": 3}
CT_GROUPS = {
    "salts": ("CT01", "CT02", "CT03", "CT04", "CT05", "CT08"),
    "highq": ("QQ09", "QP09", "QP10", "QP16"),
    "hl_all": ("HLA",),
}
HL_BONDS = (1.9, 2.2)


def _csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _selection(value):
    first, _, last = str(value).partition("-")
    return slice(int(first) - 1, int(last or first))


def _predict(model, paths, hl_refs):
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

    predictions = {}
    for path in paths:
        frames = read(path, index=":", format="extxyz")
        first = frames[0]
        info = first.info
        spin = info.get("spin", 1)
        hl = hl_refs.get(path.stem)
        try:
            if hl is None:
                offset = sum(
                    evaluate(
                        first[_selection(info[f"selection_{part}"])],
                        info[f"charge_{part}"],
                        spin,
                    )
                    for part in ("a", "b")
                )
            else:
                molecule = first[_selection(info["selection_mol"])]
                spectator = _selection(info["selection_spec"]).start
                offset = evaluate(molecule, hl["mol_charge"], hl["mol_mult"])
                offset += hl["n_spec"] * evaluate(
                    first[spectator : spectator + 1], hl["spec_charge"], hl["spec_mult"]
                )
        except (ValueError, RuntimeError, KeyError, NotImplementedError) as exc:
            logging.warning(
                "Electrostatics %s fragment evaluation failed: %s", path.stem, exc
            )
            continue

        for atoms in frames:
            frame = atoms.info
            frame_id = str(frame["frame"])
            try:
                energy = evaluate(atoms, frame["charge"], frame.get("spin", 1))
            except (ValueError, RuntimeError, KeyError, NotImplementedError) as exc:
                logging.warning("Electrostatics %s energy failed: %s", frame_id, exc)
                continue
            force = None
            if hl is not None:
                try:
                    force = float(
                        atoms.get_forces()[_selection(frame["selection_mol"]).start, 0]
                    )
                except (ValueError, RuntimeError, KeyError, NotImplementedError) as exc:
                    logging.warning("Electrostatics %s force failed: %s", frame_id, exc)
            predictions[path.stem, frame_id] = (
                (energy - offset) * units.mol / units.kcal,
                force,
                frame,
            )
    return predictions


def _tail_metrics(reference, predictions):
    points = defaultdict(dict)
    for (system, _), (energy, _, info) in predictions.items():
        if str(info["grid"]).startswith("radial") and float(info["R"]) > 10:
            points[system, info["orient"]][float(info["R"])] = energy

    distances = defaultdict(list)
    for row in _csv(reference / "tail_points.csv"):
        distances[row["system"], row["orientation"]].append(float(row["R"]))

    summary = {}
    fits = _csv(reference / "tail_fits.csv")
    for category, power in TAIL_POWERS.items():
        series = [row for row in fits if row["category"] == category]
        evaluated = passed = sign_errors = 0
        errors = []
        for row in series:
            key = row["system"], row["orientation"]
            r = distances[key]
            values = points[key]
            if len(r) < 3 or any(distance not in values for distance in r):
                continue
            evaluated += 1
            energy = np.array([values[distance] for distance in r])
            x = np.column_stack([np.asarray(r) ** -power, np.ones(len(r))])
            coefficients = np.linalg.lstsq(x, energy, rcond=None)[0]
            variance = np.sum((energy - energy.mean()) ** 2) if np.ptp(energy) else 0.0
            if (
                variance > 0
                and 1 - np.sum((energy - x @ coefficients) ** 2) / variance > 0.95
            ):
                passed += 1
            model_a = float(coefficients[0]) if variance > 0 else 0.0
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


def _curve_metrics(reference, predictions):
    ct_rows = [
        row for row in _csv(reference / "ct_points.csv") if row["benchmark"] == "ct"
    ]
    hl_rows = [
        row
        for row in _csv(reference / "hl.csv")
        if row["system"] == "HLA" and row["orientation"] in {"r=1.9", "r=2.2"}
    ]
    curves = defaultdict(list)
    for row in ct_rows + hl_rows:
        branch = row["orientation"] if row["system"] == "HLA" else ""
        curves[row["system"], branch].append(row)

    errors = defaultdict(list)
    changes = defaultdict(list)
    for (system, _), rows in curves.items():
        rows.sort(key=lambda row: float(row["R"]))
        keys = [(system, row["frame_id"]) for row in rows]
        if len(rows) < 2 or any(key not in predictions for key in keys):
            continue
        model = np.array([predictions[key][0] for key in keys])
        ref = np.array([float(row["E_int_kcal"]) for row in rows])
        errors[system].extend(abs((model - model[-1]) - (ref - ref[-1])))
        changes[system].extend(abs(ref - ref[-1]))

    # Keep HLA comparable by requiring both frozen branches.
    if any((row["system"], row["frame_id"]) not in predictions for row in hl_rows):
        errors.pop("HLA", None)

    maes = {system: float(np.mean(values)) for system, values in errors.items()}
    relative = [
        maes[system] / float(np.mean(changes[system]))
        for system in maes
        if np.mean(changes[system]) > 0.1
    ]

    groups = {}
    for group, systems in CT_GROUPS.items():
        values = [maes[system] for system in systems if system in maes]
        groups[group] = {
            "mae_kcal_mol": float(np.mean(values)) if values else None,
            "n_evaluated": len(values),
            "n_systems": len(systems),
        }

    ratios = []
    expected = 0
    for row in hl_rows:
        ref = float(row["F_bond_dft_eVA"])
        if float(row["r_bond"]) in HL_BONDS and abs(ref) > 0.05:
            expected += 1
            prediction = predictions.get(("HLA", row["frame_id"]))
            if (
                prediction is not None
                and prediction[1] is not None
                and np.isfinite(prediction[1])
            ):
                ratios.append(prediction[1] / ref)

    return {
        "groups": groups,
        "median_mae_kcal_mol": float(np.median(list(maes.values()))) if maes else None,
        "median_relative_mae": float(np.median(relative)) if relative else None,
        "restoring_force": {
            "n_expected": expected,
            "n_evaluated": len(ratios),
            "n_correct_sign": sum(ratio > 0 for ratio in ratios),
            "median_ratio": float(np.median(ratios)) if ratios else None,
        },
    }


def run_inference(model, data: Path):
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
