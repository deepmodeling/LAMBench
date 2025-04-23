import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from lambench.metrics.vishelper.metrics_calculations import MetricsCalculator


@pytest.fixture
def mock_raw_results():
    return MagicMock()


@pytest.fixture
def metrics_calculator(mock_raw_results):
    return MetricsCalculator(mock_raw_results)


def test_calculate_mean_m_bar_domain(metrics_calculator, mock_raw_results):
    model = MagicMock()
    mock_raw_results.aggregate_ood_results_for_one_model.return_value = {
        "domain1": 0.8,
        "domain2": 0.9,
    }
    result = metrics_calculator.calculate_mean_m_bar_domain(model)
    np.testing.assert_almost_equal(result, 0.85)


def test_convert_metric_to_score_minmax(metrics_calculator):
    metric_dict = {"model1": 0.2, "model2": 0.8, "model3": 0.5}
    result = metrics_calculator.convert_metric_to_score(metric_dict, method="minmax")
    expected = {"model1": 1.0, "model2": 0.0, "model3": 0.5}
    for key in result:
        np.testing.assert_almost_equal(result[key], expected[key])


def test_convert_metric_to_score_log(metrics_calculator):
    metric_dict = {"model1": 0.2, "model2": 0.8, "model3": 0.5}
    result = metrics_calculator.convert_metric_to_score(metric_dict, method="-log")
    assert result["model1"] > result["model3"] > result["model2"]


def test_calculate_generalizability_ood_error_metric(
    metrics_calculator, mock_raw_results
):
    mock_raw_results.aggregate_ood_results.return_value = {
        "model1": {"domain1": 0.8, "domain2": 0.9},
        "model2": {"domain1": 0.7, "domain2": 0.6},
    }
    result = metrics_calculator.calculate_generalizability_ood_error_metric()
    np.testing.assert_almost_equal(result["model1"], 0.85)
    np.testing.assert_almost_equal(result["model2"], 0.65)


def test_calculate_stability_results(metrics_calculator, mock_raw_results):
    mock_raw_results.fetch_stability_results.return_value = {
        "model1": {
            "system1": {"steps": 10000, "slope": 5e-4},
            "system2": {"steps": 10000, "slope": 5e-3},
        },  # log10[1], log10[10] --> 0, 1
        "model2": {
            "system1": {"steps": 10000, "slope": 0.05},
            "system2": {"steps": None, "slope": None},
        },  # log10[100], 5 --> 2, 5
    }
    result = metrics_calculator.calculate_stability_results()
    np.testing.assert_almost_equal(result["model1"], 0.5)
    np.testing.assert_almost_equal(result["model2"], 3.5)


def test_calculate_efficiency_results(metrics_calculator, mock_raw_results):
    mock_raw_results.fetch_inference_efficiency_results.return_value = {
        "model1": {"average_time": 0.5},
        "model2": {"average_time": 0.8},
    }
    result = metrics_calculator.calculate_efficiency_results()
    np.testing.assert_almost_equal(result["model1"], 200)
    np.testing.assert_almost_equal(result["model2"], 125)


def test_summarize_final_rankings(metrics_calculator):
    metrics_calculator.calculate_generalizability_ood_error_metric = MagicMock(
        return_value={"model1": 0.8, "model2": 0.6}
    )
    metrics_calculator.calculate_generalizability_downstream_score = MagicMock(
        return_value={"model1": 0.4, "model2": 0.3}
    )
    metrics_calculator.calculate_efficiency_results = MagicMock(
        return_value={"model1": 0.9, "model2": 0.7}
    )
    metrics_calculator.calculate_stability_results = MagicMock(
        return_value={"model1": 0.2, "model2": 0.5}
    )
    result = metrics_calculator.summarize_final_rankings()
    assert result is not None
    assert result.iloc[0]["Model"] == "model2"
    assert result.iloc[1]["Model"] == "model1"


def test_calculate_generalizability_downstream_score(
    metrics_calculator,
    mock_raw_results,
):
    mock_raw_results.fetch_downstream_results.return_value = pd.DataFrame(
        {
            "phonon_mdr::mae_entropy": [45.6, 24.5],
            "phonon_mdr::mae_max_freq": [58.9, 51.2],
            "phonon_mdr::success_rate": [1.0, 0.9],
            "phonon_mdr::mae_free_energy": [18.1, 14.3],
            "phonon_mdr::mae_heat_capacity": [12.8, 7.2],
        },
        index=["model1", "model2"],
    )  # Add index for models

    """
    step 1: Calculate M_bar_i for each model using dummy values
     dummy: {"mae_entropy":764.8, "mae_max_freq":1188.3, "mae_free_energy":125.1, "mae_heat_capacity":547.4}
    ===>
    pd.DataFrame({
        "phonon_mdr::mae_entropy": [45.6/764.8, 24.5/764.8],
        "phonon_mdr::mae_max_freq": [58.9/1188.3, 51.2/1188.3],
        "phonon_mdr::success_rate": [1.0, 0.9],
        "phonon_mdr::mae_free_energy": [18.1/125.1, 14.3/125.1],
        "phonon_mdr::mae_heat_capacity": [12.8/547.4, 7.2/547.4],
    }, index=["model1", "model2"])

    Step 2: Penalize with success rate for phonon_mdr
    ===>
    pd.DataFrame({
        "phonon_mdr::mae_entropy": [45.6/764.8, 24.5/764.8/0.9],
        "phonon_mdr::mae_max_freq": [58.9/1188.3, 51.2/1188.3/0.9],
        "phonon_mdr::success_rate": [1.0, 0.9],
        "phonon_mdr::mae_free_energy": [18.1/125.1, 14.3/125.1/0.9],
        "phonon_mdr::mae_heat_capacity": [12.8/547.4, 7.2/547.4/0.9],
    }, index=["model1", "model2"])

    Step 3: Calculate M_bar_domain by aggregating the results in each domain

             Inorganic Materials
    model1   0.069314
    model2   0.056273
    """

    with patch(
        "lambench.metrics.vishelper.metrics_calculations.DOWNSTREAM_TASK_METRICS",
        {
            "phonon_mdr": {
                "domain": "Inorganic Materials",
                "metrics": [
                    "mae_entropy",
                    "mae_max_freq",
                    "mae_free_energy",
                    "mae_heat_capacity",
                ],
                "penalty": "success_rate",
                "dummy": {
                    "mae_entropy": 764.8,
                    "mae_max_freq": 1188.3,
                    "mae_free_energy": 125.1,
                    "mae_heat_capacity": 547.4,
                },
            }
        },
    ):
        result = metrics_calculator.calculate_generalizability_downstream_score()
    np.testing.assert_almost_equal(result["model1"], 0.069314, decimal=5)
    np.testing.assert_almost_equal(result["model2"], 0.056273, decimal=5)
