import numpy as np
from lambench.metrics.utils import (
    filter_generalizability_force_field_results,
    aggregated_inference_efficiency_results,
    get_domain_to_direct_task_mapping,
)


def test_filter_generalizability_force_field_results():
    task_result = {
        "energy_rmse": 0.2,
        "force_rmse": 0.3,
        "virial_rmse": 0.4,
    }
    config = {
        "energy_weight": 1.0,
        "force_weight": 0.5,
        "virial_weight": None,
        "energy_std": 1.0,
        "force_std": 1.0,
        "virial_std": 1.0,
    }
    res = filter_generalizability_force_field_results(task_result, config)
    assert np.isclose(res["energy_rmse"], np.log(0.2))
    assert np.isclose(res["force_rmse"], np.log(0.3) * 0.5)
    assert res["virial_rmse"] is None
    assert task_result["virial_rmse"] is None


def test_filter_generalizability_force_field_results_normalize():
    task_result = {
        "energy_rmse": 0.2,
        "force_rmse": 0.4,
    }
    config = {
        "energy_weight": 1.0,
        "force_weight": 0.5,
        "energy_std": 0.1,
        "force_std": 0.2,
    }
    res = filter_generalizability_force_field_results(task_result, config, normalize=True)
    assert np.isclose(res["energy_rmse"], 0.0)
    assert np.isclose(res["force_rmse"], 0.0)


def test_aggregated_inference_efficiency_results_complete():
    data = {
        "sys1": {"average_time": 0.1, "std_time": 0.01, "success_rate": 1.0},
        "sys2": {"average_time": 0.2, "std_time": 0.04, "success_rate": 0.9},
    }
    res = aggregated_inference_efficiency_results(data)
    expected_std = np.sqrt((0.01 ** 2 + 0.04 ** 2) / 2)
    assert np.isclose(res["average_time"], 0.15)
    assert np.isclose(res["standard_deviation"], expected_std)
    assert np.isclose(res["success_rate"], 0.95)


def test_aggregated_inference_efficiency_results_incomplete():
    data = {
        "sys1": {"average_time": None, "std_time": 0.01, "success_rate": 1.0},
        "sys2": {"average_time": 0.2, "std_time": 0.04, "success_rate": 0.9},
    }
    res = aggregated_inference_efficiency_results(data)
    assert res == {"average_time": None, "std_time": None, "success_rate": 0.0}


def test_get_domain_to_direct_task_mapping():
    config = {
        "task1": {"domain": "A"},
        "task2": {"domain": "B"},
        "task3": {"domain": "A"},
    }
    res = get_domain_to_direct_task_mapping(config)
    assert set(res["A"]) == {"task1", "task3"}
    assert set(res["B"]) == {"task2"}
