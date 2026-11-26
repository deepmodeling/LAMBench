"""Aggregate stored electrostatics measurements for the leaderboard export."""

import numpy as np

CT_GROUPS = {
    "salts": ("CT01", "CT02", "CT03", "CT04", "CT05", "CT08"),
    "highq": ("QQ09", "QP09", "QP10", "QP16"),
    "hl_all": ("HLA",),
}
TAIL_MIN_R2 = 0.95
MIN_RELATIVE_ENERGY = 0.1


def aggregate_electrostatics(results: dict) -> dict:
    # Earlier records already contain the export summary.
    if "tail_fits" not in results:
        return results

    power_law = {}
    for category, fits in results["tail_fits"].items():
        evaluated = [fit for fit in fits if fit["coefficient"] is not None]
        errors = []
        for fit in evaluated:
            coefficient = fit["coefficient"]
            reference = fit["reference_coefficient"]
            if np.sign(coefficient) == np.sign(reference):
                errors.append(abs(1 - coefficient / reference))

        power_law[category] = {
            "n_series": len(fits),
            "n_evaluated": len(evaluated),
            "n_pass_r2": sum(
                fit["r2"] is not None and fit["r2"] > TAIL_MIN_R2 for fit in evaluated
            ),
            "sign_errors": len(evaluated) - len(errors),
            "n_eps": len(errors),
            "median_abs_eps": float(np.median(errors)) if errors else None,
            "mean_abs_eps": float(np.mean(errors)) if errors else None,
            "max_abs_eps": max(errors) if errors else None,
        }

    curves = results["aligned_energy_errors"]
    maes = [curve["mae_kcal_mol"] for curve in curves.values()]
    relative = [
        curve["mae_kcal_mol"] / curve["reference_change_kcal_mol"]
        for curve in curves.values()
        if curve["reference_change_kcal_mol"] > MIN_RELATIVE_ENERGY
    ]
    groups = {}
    for group, systems in CT_GROUPS.items():
        values = [
            curves[system]["mae_kcal_mol"] for system in systems if system in curves
        ]
        groups[group] = {
            "mae_kcal_mol": float(np.mean(values)) if values else None,
            "n_evaluated": len(values),
            "n_systems": len(systems),
        }

    forces = results["restoring_forces"]
    ratios = [
        force["predicted_ev_angstrom"] / force["reference_ev_angstrom"]
        for force in forces.values()
        if force["predicted_ev_angstrom"] is not None
    ]
    return {
        "power_law": power_law,
        "charge_transfer_and_halogen_localization": {
            "groups": groups,
            "median_mae_kcal_mol": float(np.median(maes)) if maes else None,
            "median_relative_mae": float(np.median(relative)) if relative else None,
            "restoring_force": {
                "n_expected": len(forces),
                "n_evaluated": len(ratios),
                "n_correct_sign": sum(ratio > 0 for ratio in ratios),
                "median_ratio": float(np.median(ratios)) if ratios else None,
            },
        },
    }
