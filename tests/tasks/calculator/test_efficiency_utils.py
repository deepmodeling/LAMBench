from lambench.tasks.calculator.inference_efficiency.efficiency_utils import (
    find_even_factors,
    binary_search_max_natoms,
)
import pytest
import numpy as np
from ase.atoms import Atoms
from unittest.mock import MagicMock

OOM_TEST_ATOM = Atoms(
    symbols="Mg",
    pbc=True,
    cell=[
        [-2.244256, -2.244256, 0.0],
        [-2.244256, 0.0, -2.244256],
        [0.0, -2.244256, -2.244256],
    ],
    positions=[
        [0, 0, 0],
    ],
)  # mp-1056702


@pytest.mark.parametrize(
    "num, expected",
    [
        (27, (3, 3, 3)),  # Perfect cube
        (13, (1, 1, 13)),  # Prime number
        (16, (2, 2, 4)),  # Even number
        (728, (7, 8, 13)),  # Large number
    ],
)
def test_find_even_factors(num, expected):
    result = find_even_factors(num)
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "threshold, max_natoms",
    [(1999, 1000), (247, 247), (121, 121), (100, 100), (38, 38), (31, 31)],
)
def test_binary_search_max_natoms(threshold, max_natoms):
    def mock_get_potential_energy(atoms=None):
        if len(atoms) > threshold:
            raise MemoryError("OOM: Too many atoms!")
        return np.random.rand()

    mock_model = MagicMock()
    mock_model.calc = MagicMock()
    mock_model.calc.get_potential_energy.side_effect = mock_get_potential_energy

    result = binary_search_max_natoms(mock_model, OOM_TEST_ATOM)
    assert result == max_natoms, f"Expected {max_natoms}, got {result}"
