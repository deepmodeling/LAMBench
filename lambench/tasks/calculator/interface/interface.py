import pandas as pd
from ase.io import Trajectory
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from lambench.models.ase_models import ASEModel


def run_inference(
    model: ASEModel,
    test_data: Path,
) -> dict[str, float]:
    # Conversion factor from eV/A^2 to mJ/m^2
    EV_PER_A2_TO_MJ_PER_M2 = 16021.76634

    calc = model.calc

    traj_path = test_data / "structures.traj"
    df_path = test_data / "metadata.csv"

    if not traj_path.exists() or not df_path.exists():
        logging.error(
            "Trajectory or metadata not found. Please run the conversion script first."
        )
        return {}

    df = pd.read_csv(df_path)
    traj = Trajectory(str(traj_path), "r")

    # Gather all pure structures info
    pure_df = df[df["system_type"] == "pure"]
    pure_info = {
        row["task_name"]: {
            "traj_index": row["traj_index"],
            "energy_total": row["energy_total"],
        }
        for _, row in pure_df.iterrows()
    }

    labels = []
    predictions = []

    interface_df = df[(df["system_type"] == "interface") & (df["sub1_name"].notna())]

    # Cache for energies to avoid recalculating the same pure structures and interfaces
    calc_cache = {}

    def get_energy(idx):
        if idx in calc_cache:
            return calc_cache[idx]
        try:
            atoms = traj[idx].copy()
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            calc_cache[idx] = energy
            return energy
        except Exception as e:
            logging.error(f"Error calculating energy for index {idx}: {e}")
            calc_cache[idx] = None
            return None

    for _, row in tqdm(interface_df.iterrows(), total=len(interface_df)):
        area = row["area"]
        traj_idx = row["traj_index"]
        e_label_tot = row["energy_total"]

        sub1_name = row["sub1_name"]
        sub1_mult = row["sub1_mult"]
        sub2_name = row["sub2_name"]
        sub2_mult = row["sub2_mult"]

        if sub1_name not in pure_info or sub2_name not in pure_info:
            continue

        p1 = pure_info[sub1_name]
        p2 = pure_info[sub2_name]

        # Calculate Label
        e_int_label_eV = (
            e_label_tot
            - sub1_mult * p1["energy_total"]
            - sub2_mult * p2["energy_total"]
        )
        e_int_label = (
            (e_int_label_eV / area) * EV_PER_A2_TO_MJ_PER_M2 / 2
        )  # divide by 2 to get the interfacial energy per interface instead of per supercell

        # Calculate Prediction
        e_calc_tot = get_energy(traj_idx)
        e_calc_p1 = get_energy(p1["traj_index"])
        e_calc_p2 = get_energy(p2["traj_index"])

        if e_calc_tot is None or e_calc_p1 is None or e_calc_p2 is None:
            continue

        e_int_pred_eV = e_calc_tot - sub1_mult * e_calc_p1 - sub2_mult * e_calc_p2
        e_int_pred = (e_int_pred_eV / area) * EV_PER_A2_TO_MJ_PER_M2 / 2
        labels.append(e_int_label)
        predictions.append(e_int_pred)

    if not labels:
        logging.warning("No interfaces were successfully evaluated.")
        return {}

    return {
        "MAE": mean_absolute_error(labels, predictions),  # mJ/m^2
        "RMSE": root_mean_squared_error(labels, predictions),  # mJ/m^2
    }
