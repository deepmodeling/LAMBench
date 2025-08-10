from lambench.models.ase_models import ASEModel
from lambench.tasks.calculator.inference_efficiency.efficiency_utils import (
    binary_search_max_natoms,
    get_efv,
    find_even_factors,
)
from ase.io import read
import logging
import time
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_inference(
    model: ASEModel,
    test_data: Path,
    warmup_ratio: float,
    natoms_upper_limit: int = 1000,
) -> dict[str, dict[str, float]]:
    """
    Inference for all trajectories, return average time and success rate for each system.
    """
    results = {}
    trajs = list(test_data.rglob("*.traj"))
    for traj in trajs:
        system_name = traj.name
        try:
            system_result = run_one_inference(
                model, traj, warmup_ratio, natoms_upper_limit
            )
            average_time = system_result["average_time"]
            std_time = system_result["std_time"]
            success_rate = system_result["success_rate"]
            results[system_name] = {
                "average_time": average_time,
                "std_time": std_time,
                "success_rate": success_rate,
            }
            logging.info(
                f"Inference completed for system {system_name} with average time {average_time} s and success rate {success_rate:.2f}%"
            )
        except Exception as e:
            logging.error(f"Error in inference for system {system_name}: {e}")
            results[system_name] = {
                "average_time": None,
                "std_time": None,
                "success_rate": 0.0,
            }
    return results


def run_one_inference(
    model: ASEModel,
    test_traj: Path,
    warmup_ratio: float,
    natoms_upper_limit: int,
) -> dict[str, float]:
    """
    Infer for one trajectory, return averaged time and success rate, starting timing at warmup_ratio.
    """
    test_atoms = read(test_traj, ":")
    start_index = int(len(test_atoms) * warmup_ratio)
    valid_steps = 0
    successful_inferences = 0
    total_inferences = len(test_atoms)

    efficiency = []
    for i, atoms in enumerate(test_atoms):
        # find maximum allowed natoms
        max_natoms = binary_search_max_natoms(model, atoms, natoms_upper_limit)
        # on-the-fly expand atoms
        scaling_factor = np.int32(np.floor(max_natoms / len(atoms)))
        a, b, c = find_even_factors(scaling_factor)  # a,b,c is in ascending order

        cell_length = atoms.get_cell_lengths_and_angles()[:3]
        scaling_index = np.argsort(
            cell_length
        ).tolist()  # sort cell length by ascending order
        # expand atoms, repeat shorter cell axis with larger scaling factor
        a, b, c = (
            [c, b, a][scaling_index[0]],
            [c, b, a][scaling_index[1]],
            [c, b, a][scaling_index[2]],
        )
        atoms = atoms.repeat((a, b, c))
        model.calc.reset()  # reset calculator to prevent cached results
        atoms.calc = model.calc
        n_atoms = len(atoms)
        start = time.time()
        try:
            get_efv(atoms)
            successful_inferences += 1
        except Exception as e:
            logging.error(f"Error in inference for {str(atoms.symbols)}: {e}")
            continue

        end = time.time()
        elapsed_time = end - start

        if i >= start_index:
            efficiency.append(
                elapsed_time / n_atoms * 1e6
            )  # inference efficiency in µs/atom
            valid_steps += 1

    if valid_steps > 0:
        average_efficiency = np.mean(efficiency)
        std_efficiency = np.std(efficiency)
    else:
        average_efficiency = None
        std_efficiency = None

    if total_inferences > 0:
        success_rate = (successful_inferences / total_inferences) * 100
    else:
        success_rate = 0.0

    return {
        "average_time": average_efficiency,
        "std_time": std_efficiency,
        "success_rate": success_rate,
    }
