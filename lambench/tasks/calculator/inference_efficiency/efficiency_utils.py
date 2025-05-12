from ase.atoms import Atoms
from lambench.models.ase_models import ASEModel
import numpy as np
import math


def get_efv(atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Perform force field prediction for one system, return energy, forces and stress.
    """
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


def catch_oom_error(atoms: Atoms) -> bool:
    """
    Catch OOM error when running inference.
    """
    try:
        get_efv(atoms)
        return False
    except Exception as e:
        if "out of memory" in str(e) or "OOM" in str(e):
            return True
        else:
            return False


def get_divisors(num: int) -> list[int]:
    divisors = set()
    for i in range(1, int(math.isqrt(num)) + 1):
        if num % i == 0:
            divisors.add(i)
            divisors.add(num // i)
    return sorted(divisors)


def find_even_factors(num: int) -> tuple[int, int, int]:
    """
    Find three factors of a number that are as evenly distributed as possible.
    The function returns a tuple of three factors (a, b, c) such that a * b * c = num.
    The factors are sorted in ascending order (a <= b <= c).
    """
    divisors = get_divisors(num)
    best = None
    min_spread = float("inf")

    for a in divisors:
        num_div_a = num // a
        divisors_b = get_divisors(num_div_a)

        # Since a <= b <= c, no need to consider b < a
        for b in divisors_b:
            if b < a:
                continue
            c = num_div_a // b
            if a * b * c == num:
                factors = [a, b, c]
                spread = max(factors) - min(factors)
                if spread < min_spread:
                    min_spread = spread
                    best = (a, b, c)
                    if spread == 0:  # Perfect distribution found
                        return best
    return best


def binary_search_max_natoms(
    model: ASEModel, atoms: Atoms, upper_limit: int = 1000, max_iterations: int = 15
) -> int:
    """
    Binary search for the maximum number of atoms that can be processed by the model.

    """
    low, high, iteration = 1, upper_limit, 0
    while low < high and iteration < max_iterations:
        mid = (low + high + 1) // 2
        scaling_factor = np.int32(np.ceil(mid / len(atoms)))
        scaled_atoms = atoms.copy()
        a, b, c = find_even_factors(scaling_factor)
        scaled_atoms = scaled_atoms.repeat((a, b, c))
        scaled_atoms.calc = model.calc
        if catch_oom_error(scaled_atoms):
            high = mid - 1
        else:
            low = mid
        iteration += 1
    return low
