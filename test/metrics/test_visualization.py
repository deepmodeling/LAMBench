from lambench.metrics.visualization import aggregate_domain_results_for_one_model
from lambench.models.dp_models import DPModel
import logging
import numpy as np


def test_aggregate_domain_results_for_one_model(
    mock_direct_predict_query, valid_model_data, caplog
):
    model = DPModel(**valid_model_data)
    model.model_name = "test_dp"
    model.show_direct_task = True
    model.show_finetune_task = False
    model.show_calculator_task = False
    result = aggregate_domain_results_for_one_model(model)
    np.testing.assert_almost_equal(result["Small Molecules"], 0.340232, decimal=5)
    np.testing.assert_almost_equal(
        result["Inorganic Materials"], 0.124034, decimal=5
    )
    assert result["Catalysis"] is None
    with caplog.at_level(logging.WARNING):
        assert (
            "Expect one record for test_dp and CGM_MLP_NC2023, but got 0" in caplog.text
        )
