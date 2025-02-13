from lambench.metrics.post_process import (
    process_results_for_one_model,
    DIRECT_TASK_WEIGHTS,
    exp_average,
)
from lambench.models.dp_models import DPModel
import logging
import numpy as np


def test_process_results_for_one_model(
    mock_direct_predict_query, valid_model_data, caplog
):
    model = DPModel(**valid_model_data)
    model.model_name = "test_dp"
    model.show_direct_task = True
    model.show_finetune_task = False
    model.show_calculator_task = False
    result = process_results_for_one_model(model)

    assert DIRECT_TASK_WEIGHTS.keys() - result["direct_task_results"].keys() == {
        "Collision",
        "CGM_MLP_NC2023",
        "H_nature_2022",
    }
    with caplog.at_level(logging.WARNING):
        assert (
            "Weighted results for test_dp are marked as None due to missing tasks: "
            in caplog.text
        )
    assert result["direct_task_results"]["Weighted"] is None
    assert result["direct_task_results"]["ANI"]["energy_rmse"] == 0.467693
    assert result["direct_task_results"]["WBM_downsampled"]["force_rmse"] is None


def test_average():
    log_results = [
        {
            "force_rmse": np.log(0.437947),
        },
        {
            "force_rmse": np.log(0.30147),
        },
        {
            "force_rmse": np.log(0.345338),
        },
        {
            "force_rmse": np.log(0.19703),
        },
        {
            "force_rmse": np.log(0.0924748),
        },
        {
            "force_rmse": np.log(0.221132),
        },
        {
            "force_rmse": np.log(0.112146),
        },
        {
            "force_rmse": np.log(0.266102),
        },
        {
            "force_rmse": np.log(0.0584197),
        },
        {
            "force_rmse": np.log(0.13488),
        },
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
    np.testing.assert_allclose(exp_average(log_results)["force_rmse"], 0.1866048)
