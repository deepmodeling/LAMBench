from pathlib import Path

from ase import Atoms
import dpdata
import sklearn.metrics as metrics
import pandas as pd
from lambench.models.ase_models import ASEModel
from tqdm import tqdm


def run_torsionnet(
    model: ASEModel,
    test_data: Path,
) -> dict:
    metric = {}
    result = {}  # predicted energy
    label = {}  # label energy

    # 500 fragments from Torsionnet500 dataset
    for fragment in tqdm(test_data.iterdir()):
        if not fragment.is_dir():
            continue
        result[fragment.name] = []
        label[fragment.name] = []
        # 24 conformations for each fragment
        for frame in dpdata.LabeledSystem(file_name=fragment, fmt="deepmd/raw"):
            assert len(frame) == 1
            atoms: Atoms = frame.to_ase_structure()[0]  # type: ignore
            atoms.calc = model.calc
            pred_energy = atoms.get_potential_energy() * 23.0609  # eV to kcal/mol
            result[fragment.name].append(pred_energy)
            label_energy = frame.data["energies"][0]
            label[fragment.name].append(label_energy)
    result_df = pd.DataFrame.from_dict(result, orient="index")
    label_df = pd.DataFrame.from_dict(label, orient="index")

    # barrier height MAE (barrier height is the max energy - min energy of the fragment)
    result_barrier = result_df.max(axis=1) - result_df.min(axis=1)
    label_barrier = label_df.max(axis=1) - label_df.min(axis=1)
    metric["MAEB"] = metrics.mean_absolute_error(
        label_barrier,
        result_barrier,
    )

    # number of molecules with error of a barrier higher more than 1 kcal/mol
    barrier_diff = (result_barrier - label_barrier).abs()
    print(barrier_diff.mean(axis=None))
    metric["NABH_h"] = sum(barrier_diff > 1)

    # normalize the energies
    result_df = result_df.sub(result_df.min(axis=1), axis=0)
    ## label is already normalized
    # label_df = label_df.sub(label_df.min(axis=1), axis=0)
    assert label_df.min(axis=1).max(axis=0) == 0
    metric["MAE"] = metrics.mean_absolute_error(label_df, result_df)
    metric["RMSE"] = metrics.root_mean_squared_error(label_df, result_df)
    return metric
