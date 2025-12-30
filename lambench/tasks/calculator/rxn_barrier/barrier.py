"""
The test data is retrieved from:

	@misc{liang2025gold,
      title={Gold-Standard Chemical Database 137 (GSCDB137): A diverse set of accurate energy differences for assessing and developing density functionals}, 
      author={Jiashu Liang and Martin Head-Gordon},
      year={2025},
      eprint={2508.13468},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.13468}, 
}

https://github.com/JiashuLiang/GSCDB

Only the BH876 dataset is used.

"""

from ase.io import read
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from pathlib import Path
from lambench.models.ase_models import ASEModel
import logging




def run_inference(
    model: ASEModel,
    test_data: Path,
) -> dict[str, float]:

    lookup_table = pd.read_csv(test_data / "lookup_table.csv")
    lookup_table.reset_index(inplace=True)
    stoichiometry = pd.read_csv(test_data / "stoichiometry.csv")
    traj = read(test_data / "BH876.traj", ":")

    EV_TO_KCAL = 23.06092234465
    HARTREE_TO_KCAL = 627.50947406

    preds = []
    labels = []
    success = len(stoichiometry)

    calc =  model.calc

    for i, row in tqdm(stoichiometry.iterrows()):
        try:
            reactions = row["Stoichiometry"].split(",")
            num_species = len(reactions) // 2 
            pred = 0
            for i in range(num_species):
                stoi = float(reactions[2*i])
                reactant = reactions[2*i+1] 
                structure_index = lookup_table[lookup_table["ID"] == reactant].index.values[0]
                atoms = traj[structure_index]
                atoms.info.update({"fparam": np.array([atoms.info["charge"], atoms.info["spin"]])})
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                pred += stoi * energy
            preds.append(pred * EV_TO_KCAL)
            labels.append(row["Reference"] * HARTREE_TO_KCAL)
        except:
            logging.warning(f"Failed to calculate reaction energy for reaction: {row['Stoichiometry']}")
            success -= 1
    

    return {
        "MAE": mean_absolute_error(labels, preds),  # kcal/mol
        "RMSE": root_mean_squared_error(labels, preds),  # kcal/mol
        "success_rate": success / len(stoichiometry),
    }