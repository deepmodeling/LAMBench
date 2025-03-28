from lambench.tasks.calculator.nve_md.nve_md import (
    nve_simulation_single,
    run_md_nve_simulation,
)
from lambench.metrics.utils import aggregated_nve_md_results
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from lambench.models.ase_models import ASEModel
import numpy as np


@pytest.fixture
def setup_testing_data():
    """Fixture to provide testing data for NVE simulation."""
    return Atoms(
        "H2O", positions=[(0.0, 0.0, 0.0), (0.0, 0.757, 0.587), (0.0, -0.757, 0.587)]
    )  # Example system


@pytest.fixture
def setup_calculator():
    """Fixture to provide an ASE calculator (EMT)."""
    return EMT()


@pytest.fixture
def setup_model(setup_calculator):
    """Fixture to provide an ASE model."""
    ase_models = ASEModel(
        model_family="TEST",
        model_type="ASE",
        model_name="",
        model_metadata={
            "pretty_name": "test",
            "num_parameters": 1000,
            "packages": {"torch": "2.0.0"},
        },
        virtualenv="",
    )
    ase_models.calc = setup_calculator
    return ase_models


def test_nve_simulation_metrics(setup_testing_data, setup_calculator):
    """Test NVE simulation metrics for std, and steps."""
    result = nve_simulation_single(
        setup_testing_data,
        setup_calculator,
        timestep=1.0,
        num_steps=100,
        temperature_K=300,
    )

    assert result["steps"] > 0, "Steps should be greater than zero."
    assert result["simulation_time"] > 0, "Simulation time should be greater than zero."
    assert isinstance(result["slope"], float), "Slope should be a float."


def test_nve_simulation_crash_handling(setup_testing_data, setup_calculator):
    """Test crash handling by simulating an intentional crash."""
    atoms = setup_testing_data

    def faulty_calculator(a):
        """A faulty calculator to simulate a crash."""
        raise RuntimeError("Intentional crash for testing.")

    res = nve_simulation_single(
        atoms, faulty_calculator, timestep=1.0, num_steps=100, temperature_K=300
    )
    assert res["steps"] == 0, "Simulation should crash."
    assert res["slope"] is None, f"Slope should be NaN, got {res['slope']}."


def test_run_md_nve_simulation(setup_testing_data, setup_model):
    """Test running NVE simulation for a model."""
    result = run_md_nve_simulation(
        setup_model,
        timestep=1.0,
        num_steps=100,
        temperature_K=300,
        test_data=[setup_testing_data],
    )
    assert isinstance(result, dict), "Result should be a dictionary."
    assert set(result.keys()) == {
        "H2O",
    }, "Result should have keys 'H2O'."
    assert (
        set(result["H2O"].keys())
        == {
            "simulation_time",
            "steps",
            "slope",
            "std",
            "momenta_diff",
        }
    ), "Result should have keys 'simulation_time', 'steps', 'std', 'slope', 'momenta_diff'."
    assert result["H2O"]["steps"] > 0, "Steps should be greater than zero."
    assert isinstance(
        result["H2O"]["simulation_time"], float
    ), "Simulation time should be a float."
    assert isinstance(result["H2O"]["slope"], float), "Slope should be a float."
    assert isinstance(
        result["H2O"]["momenta_diff"], float
    ), "Momenta diff should be a float."


def test_run_md_nve_simulation_crash_handling(setup_model, setup_testing_data):
    """Test crash handling by simulating an intentional crash."""

    def faulty_calculator(a):
        """A faulty calculator to simulate a crash."""
        raise RuntimeError("Intentional crash for testing.")

    setup_model.calc = faulty_calculator
    result = run_md_nve_simulation(
        setup_model,
        timestep=1.0,
        num_steps=100,
        temperature_K=300,
        test_data=[setup_testing_data],
    )
    assert isinstance(result, dict), "Result should be a dictionary."
    assert set(result.keys()) == {
        "H2O",
    }, "Result should have keys 'H2O'."
    assert (
        set(result["H2O"].keys())
        == {
            "simulation_time",
            "steps",
            "slope",
            "std",
            "momenta_diff",
        }
    ), "Result should have keys 'simulation_time', 'steps', 'std', 'slope', 'momenta_diff'."


def test_aggreated_results():
    """Test aggregation of results."""
    results = {
        "Cs8N2": {
            "simulation_time": 128.3,
            "steps": 1000,
            "slope": None,
            "momenta_diff": 0.1,
        },
        "Gd2Si4Ni2": {
            "simulation_time": 2374.1,
            "steps": 10000,
            "slope": 4580.2,
            "momenta_diff": 200020.2,
        },
    }
    result = aggregated_nve_md_results(results)
    np.testing.assert_almost_equal(result["simulation_time"], 2374.1, decimal=3)
    assert result["steps"] == 10000, "Should skip incomplete test."
    assert result["slope"] == 4580.2, "Should skip incomplete test."
    np.testing.assert_almost_equal(result["momenta_diff"], 200020.2, decimal=3)
    assert result["success_rate"] == 0.5, "Should have 1 success."
