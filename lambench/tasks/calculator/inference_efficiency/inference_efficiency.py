from lambench.models.ase_models import ASEModel
from lambench.tasks.calculator.inference_efficiency.efficiency_utils import binary_search_max_natoms, get_efv, find_even_factors
from ase.io import read, write
from ase.atoms import Atoms
import logging
import time
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# OOM_TEST_ATOM = Atoms(
#         symbols="H",
#         pbc=True,
#         cell=[
#             [2.64963874, 0.        , 0.        ],
#             [0.        , 2.64963874, 0.        ],
#             [0.        , 0.        , 0.98049292]
#         ],
#         positions=[
#             [0.        , 0.        , 0.49024646]
#         ],
#     )
# OOM_TEST_ATOM = read(
#     "/mnt/data_nas/guomingyu/DPA_BENCHMARKS/LAMBENCH_binary_experiments/CONFs/HPt_NC2022_0_4.cif"
# )
def run_inference(
    model: ASEModel, test_data: Path, warmup_ratio: float
) -> dict[str, dict[str, float]]:
    """
    Inference for all trajectories, return average time and success rate for each system.
    """
    results = {}
    trajs = list(test_data.rglob("*.traj"))
    # find maximum allowed natoms 
    max_natoms = binary_search_max_natoms(model, OOM_TEST_ATOM)
    logging.info(f"Yielded converged n_atoms {max_natoms}")
    for traj in trajs:
        system_name = traj.name
        try:
            system_result = run_one_inference(model, traj, warmup_ratio, max_natoms)
            average_time = system_result["average_time"]
            std_time = system_result["std_time"]
            success_rate = system_result["success_rate"]
            results[system_name] = {
                "average_time": average_time,
                "std_time": std_time,
                "success_rate": success_rate,
                "max_atoms": max_natoms
            }
            logging.info(
                f"Inference completed for system {system_name} with average time {average_time} s and success rate {success_rate:.2f}%"
            )
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            logging.error(f"Error in inference for system {system_name}: {e}")
            results[system_name] = {
                "average_time": None,
                "std_time": None,
                "success_rate": 0.0,
                "max_atoms": max_natoms
            }
    return results


def run_one_inference(
    model: ASEModel, test_traj: Path, warmup_ratio: float, max_natoms: int
) -> dict[str, float]:
    """
    Infer for one trajectory, return averaged time and success rate, starting timing at warmup_ratio.
    """
    test_atoms = read(test_traj, ":")
    logging.info(f"TEST TRAJ P {test_traj.parent}")
    start_index = int(len(test_atoms) * warmup_ratio)
    valid_steps = 0
    successful_inferences = 0
    total_inferences = len(test_atoms)
    efficiency = []
    for i, atoms in enumerate(test_atoms):
        # on-the-fly expand atoms 
        scaling_factor = np.int32(np.floor(max_natoms / len(atoms)))
        while 1 in find_even_factors(scaling_factor) and scaling_factor > 1:
            scaling_factor -= 1
        a, b, c = find_even_factors(scaling_factor)
        atoms = atoms.repeat((a,b,c))
        n_atoms = len(atoms)
        atoms.calc = model.calc
        start = time.time()
        try:
            get_efv(atoms)
            successful_inferences += 1
        except Exception as e:
            if "out of memory" in str(e) or "OOM" in str(e):
                write(f"{test_traj.parent}/oom_traj_{i}.extxyz", atoms, format='extxyz', append=True)
            else:
                write(f"{test_traj.parent}/unknwon_error_{i}.extxyz", atoms, format='extxyz', append=True)
            import traceback
            logging.error(traceback.format_exc())
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
