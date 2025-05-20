from lambench.models.ase_models import ASEModel
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from ase.io import read
from sklearn.metrics import mean_absolute_error


def run_inference(
    model: ASEModel,
    test_data: Path,
) -> dict[str, float]:
    """ "
    This tests perform energy barrier prediction for 460 OOD NEB trajs in the OC20NEB dataset.

    The label.csv file contains the ground truth energy barrier Ea and dE for each trajectory.
    Each trajectory file contains 10 frames, and the first and last frames are the initial and final states.
    Using the ground truth energy label, the index of the transition state is determined.

    We then perform single point inference for the initial, final, and transition states and calculate Ea and dE.
    """
    result_df = pd.read_csv(Path(test_data, "label.csv"))
    result_df.sort_values("traj", inplace=True)
    result_df["pred_Ea"] = None
    result_df["pred_dE"] = None
    NUM_RECORDS = len(result_df)
    ERROR_THRESHOLD = 0.1  # counting Ea predtion error > 0.1 eV as failure

    for idx, row in tqdm(result_df.iterrows()):
        traj_name = row["traj"]
        data = read(test_data, f"{traj_name}.traj", ":")
        energies = [frame.get_potential_energy() for frame in data]
        barrier_idx = np.argmax(energies)

        initial, transition, final = data[0], data[barrier_idx], data[-1]
        pred_energy = []
        for atoms in [initial, transition, final]:
            pred_energy.append(atoms.get_potential_energy())
        e_a, de = (
            pred_energy[1] - pred_energy[0],
            pred_energy[-1] - pred_energy[0],
        )
        result_df.at[idx, "pred_Ea"] = e_a
        result_df.at[idx, "pred_dE"] = de

    result_df["type"] = result_df["traj"].apply(lambda x: x.split("_")[0])
    result_df["error"] = result_df.apply(
        lambda x: np.abs(x["pred_Ea"] - x["Ea"]), axis=1
    )
    type_percentages = (
        result_df[result_df["error"] > ERROR_THRESHOLD].groupby("type").size()
        / result_df.groupby("type").size()
        * 100
    )
    type_percentages = type_percentages.round(2)
    results = type_percentages.to_dict()
    type_percentages.dropna(inplace=True)
    results["success_rate"] = len(type_percentages) / NUM_RECORDS * 100
    results["average_error"] = mean_absolute_error(
        result_df["Ea"], result_df["pred_Ea"]
    )
    return results
