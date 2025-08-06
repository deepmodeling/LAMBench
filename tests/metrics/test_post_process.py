from lambench.metrics.post_process import (
    process_results_for_one_model,
    DIRECT_TASK_WEIGHTS,
    exp_average,
)
from lambench.models.dp_models import DPModel
from lambench.metrics.utils import CALCULATOR_TASKS
import logging
import numpy as np


def test_process_results_for_one_model(
    mock_direct_predict_query, mock_calculator_query, valid_model_data, caplog
):
    model = DPModel(**valid_model_data)
    model.model_name = "test_dp"
    model.show_direct_task = True
    model.show_finetune_task = False
    model.show_calculator_task = True
    result = process_results_for_one_model(model)

    assert DIRECT_TASK_WEIGHTS.keys() - result[
        "generalizability_force_field_results"
    ].keys() == {
        "Cu_MgO_catalysts"
    }
    with caplog.at_level(logging.WARNING):
        assert (
            "Weighted results for test_dp are marked as None due to missing tasks: "
            in caplog.text
        )
    assert result["generalizability_force_field_results"]["Weighted"] is None
    assert result["generalizability_force_field_results"]["ANI"]["energy_rmse"] == 467.7

    # Find differences between the calculator tasks and results
    calculator_task_differences = (
        CALCULATOR_TASKS.keys() - {"inference_efficiency", "nve_md"}
    ).symmetric_difference(result["generalizability_domain_specific_results"].keys())
    assert not calculator_task_differences, (
        f"Mismatch in calculator tasks: {calculator_task_differences}"
    )


def test_average():
    log_results = [
        {
            "force_rmse": np.log(0.0626074),
        },
        {
            "force_rmse": np.log(0.182507),
        },
        {
            "force_rmse": None,
        },
        {
            "force_rmse": np.log(0.624174),
        },
        {
            "force_rmse": np.log(0.201489),
        },
    ]
    np.testing.assert_allclose(exp_average(log_results)["force_rmse"], 0.1946998)
