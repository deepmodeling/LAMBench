from ase.io import read
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lambench.models.ase_models import ASEModel
from lambench.tasks.calculator.stacking_fault.utils import fit_pchip


EV_A2_TO_MJ_M2 = 16021.766208
NUM_POINTS = 200


def calc_one_traj(traj, label, calc):
    atoms_list = read(traj, ":")
    df = pd.read_csv(label, header=0)

    a_vector = atoms_list[0].cell[0]
    b_vector = atoms_list[0].cell[1]
    area = np.linalg.norm(np.cross(a_vector, b_vector))

    d = df["Displacement"]
    preds = []
    for atoms in atoms_list:
        atoms.calc = calc
        preds.append(atoms.get_potential_energy())
    res = pd.DataFrame({"Displacement": d.to_list(), "Energy": preds})
    res["Energy"] = (res["Energy"] - res["Energy"].min()) * EV_A2_TO_MJ_M2 / area

    _, y_smooth_label = fit_pchip(
        df, x_col="Displacement", y_col="Energy", num_points=NUM_POINTS
    )
    _, y_smooth_pred = fit_pchip(
        res, x_col="Displacement", y_col="Energy", num_points=NUM_POINTS
    )

    derivative_label = (
        (y_smooth_label[1:] - y_smooth_label[:-1])
        * (NUM_POINTS - 1)
        / max(y_smooth_label)
    )
    derivative_pred = (
        (y_smooth_pred[1:] - y_smooth_pred[:-1]) * (NUM_POINTS - 1) / max(y_smooth_pred)
    )

    return np.round(mean_absolute_error(y_smooth_label, y_smooth_pred), 4), np.round(
        mean_absolute_error(derivative_label, derivative_pred), 4
    )


def run_inference(model: ASEModel, test_data: Path) -> dict:
    calc = model.calc

    traj_files = sorted(list(test_data.rglob("*.traj")))
    label_files = sorted(list(test_data.rglob("*.csv")))

    energy_maes = []
    derivative_maes = []
    for traj_file, label_file in tqdm(
        zip(traj_files, label_files),
        total=len(traj_files),
        desc="Calculating Stacking Fault Energies",
    ):
        energy_mae, derivative_mae = calc_one_traj(traj_file, label_file, calc)
        energy_maes.append(energy_mae)
        derivative_maes.append(derivative_mae)

    return {
        "MAE_E": np.round(np.mean(energy_maes), 4),  # mJ/m²
        "MAE_dE": np.round(np.mean(derivative_maes), 4),  # 1/unit displacement
    }
