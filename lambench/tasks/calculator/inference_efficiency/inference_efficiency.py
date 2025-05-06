from lambench.models.ase_models import ASEModel
from ase.io import read
from ase.atoms import Atoms
import logging
import time
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_efv(atoms: Atoms) -> tuple[float, np.ndarray]:
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    stress = atoms.get_stress()
    v = (
        -np.array(
            [
                [stress[0], stress[5], stress[4]],
                [stress[5], stress[1], stress[3]],
                [stress[4], stress[3], stress[2]],
            ]
        )
        * atoms.get_volume()
    )
    return e, f, v


def find_maximum_scaling(model: ASEModel, base_atoms: Atoms, max_trials: int = 10) -> int:
    """
    Find the maximum expansion factor using binary search such that get_efv does not OOM.

    1. Start with factor = 1.
    2. Double the factor until get_efv succeeds or max_trials reached.
    3. If OOM occurs, perform binary search between last successful and failed factor.
    Returns the largest safe integer scaling factor.
    """
    def test_scale(factor: int) -> bool:
        # Expand cell evenly in three directions
        scaled = base_atoms.repeat((factor, factor, factor))
        scaled.calc = model.calc
        try:
            get_efv(scaled)
            return True
        except MemoryError: 
            return False
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "ResourceExhausted" in str(e):
                return False
            else:
                return True 
        except Exception: 
            return True

    # Find upper bound through exponential growth
    low, high = 1, 1
    for _ in range(max_trials):
        if test_scale(high):
            low = high
            high *= 2
        else:
            break

    # Binary search refinement
    while low + 1 < high:
        mid = (low + high) // 2
        if test_scale(mid):
            low = mid
        else:
            high = mid
    return low


def run_inference(
    model: ASEModel,
    test_data: Path,
    warmup_ratio: float,
    mode: str = "direct_loop"
) -> dict[str, dict[str, float]]:
    """
    Inference for all trajectories. Supports 'direct_loop' or 'binary_search' mode.
    """
    results = {}
    trajs = list(test_data.rglob("*.traj"))
    for traj in trajs:
        system_name = traj.name
        try:
            system_result = run_one_inference(model, traj, warmup_ratio, mode)
            results[system_name] = {
                "average_time": system_result["average_time"],
                "std_time": system_result["std_time"],
                "success_rate": system_result["success_rate"],
            }
            logging.info(
                f"Inference completed for system {system_name} with average time {system_result['average_time']} s and success rate {system_result['success_rate']:.2f}%"
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
    mode: str = "direct_loop"
) -> dict[str, float]:
    """
    Infer for one trajectory with selected mode.
    """
    test_atoms = read(test_traj, ":")
    start_index = int(len(test_atoms) * warmup_ratio)
    valid_steps = 0
    successful_inferences = 0
    total_inferences = len(test_atoms)
    efficiency = []

    # Determine maximum safe scaling factor
    scale_factor = 1
    if mode == "binary_search":
        base = test_atoms[0]
        scale_factor = find_maximum_scaling(model, base)
        logging.info(f"Maximum safe scaling factor for {test_traj.name}: {scale_factor}")

    for i, atoms in enumerate(test_atoms):
        # Apply scaling if in binary search mode
        if mode == "binary_search":
            atoms = atoms.repeat((scale_factor, scale_factor, scale_factor))
        
        atoms.calc = model.calc
        n_atoms = len(atoms)
        start = time.time()
        
        try:
            get_efv(atoms)
            successful_inferences += 1
        except Exception as e:
            logging.error(f"Error at step {i} of {test_traj.name}: {e}")
            continue

        elapsed_time = time.time() - start

        if i >= start_index:
            efficiency.append(elapsed_time / n_atoms * 1e6)
            valid_steps += 1

    # Calculate performance metrics
    avg_eff = np.mean(efficiency) if valid_steps > 0 else None
    std_eff = np.std(efficiency) if valid_steps > 0 else None
    success_rate = (successful_inferences / total_inferences * 100) if total_inferences > 0 else 0.0

    return {
        "average_time": avg_eff,
        "std_time": std_eff,
        "success_rate": success_rate,
    }