import logging
from unittest.mock import patch

from lambench.models.dp_models import DPModel
from lambench.tasks import PropertyFinetuneTask


def test_load_finetune_task(finetune_yml_data):
    for task_name, task_param in finetune_yml_data.items():
        model_name = "TEST_DP_v1"
        task = PropertyFinetuneTask(
            model_name=model_name, task_name=task_name, **task_param
        )
        for key, value in task_param.items():
            if key == "finetune_params":
                for k, v in value.items():
                    assert getattr(getattr(task, key), k) == v
            else:
                assert getattr(task, key) == value


def test_record_count_none(mock_record_count, finetune_task_data):
    mock_record_count.return_value = 0

    task = PropertyFinetuneTask(**finetune_task_data)
    result = task.exist(model_name="model1")
    assert result is False


def test_record_count_single(mock_record_count, finetune_task_data):
    mock_record_count.return_value = 1
    task = PropertyFinetuneTask(**finetune_task_data)
    result = task.exist(model_name="model1")
    assert result is True


def test_record_count_multiple(mock_record_count, finetune_task_data, caplog):
    mock_record_count.return_value = 2

    task = PropertyFinetuneTask(**finetune_task_data)
    with caplog.at_level(logging.WARNING):
        result = task.exist(model_name="model1")
        assert result is True
    mock_record_count.assert_called_once_with(
        task_name=finetune_task_data["task_name"], model_name="model1"
    )
    assert (
        f"Multiple records found for task {finetune_task_data['task_name']}"
        in caplog.text
    )


def test_run_task_existing_record(
    mock_record_count, valid_model_data, finetune_task_data, caplog
):
    mock_record_count.return_value = 1

    task = PropertyFinetuneTask(**finetune_task_data)
    model = DPModel(**valid_model_data)
    with caplog.at_level(logging.INFO):
        task.run_task(model)

    assert (
        f"TASK {finetune_task_data['task_name']} record found in database, SKIPPING."
        in caplog.text
    )
    mock_record_count.assert_called_once_with(
        task_name=finetune_task_data["task_name"], model_name="model1"
    )


def test_run_task_no_existing_record(
    mock_record_count, mock_record_insert, valid_model_data, finetune_task_data, caplog
):
    mock_record_count.return_value = 0

    # Mock the evaluate method
    model = DPModel(**valid_model_data)
    with (
        patch.object(
            DPModel, "evaluate", return_value={"property_rmse": 0.42}
        ) as mock_run_task,
        caplog.at_level(logging.INFO),
    ):
        task = PropertyFinetuneTask(**finetune_task_data)
        task.run_task(model)

    mock_run_task.assert_called_once()
    assert (
        f"TASK {finetune_task_data['task_name']}"
        + " OUTPUT: {'property_rmse': 0.42}, INSERTING."
        in caplog.text
    )
    mock_record_count.assert_called_once_with(
        task_name=finetune_task_data["task_name"], model_name="model1"
    )
