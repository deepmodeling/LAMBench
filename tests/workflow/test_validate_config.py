from lambench.tasks.base_task import BaseTask
from lambench.workflow.entrypoint import MODELS
import yaml
from lambench.models.ase_models import ASEModel
from lambench.models.dp_models import DPModel


def test_validate_model_config():
    assert MODELS.exists(), f"MODELS file {MODELS} does not exist."

    model_config_error_list = []

    with open(MODELS, "r") as f:
        model_config = yaml.safe_load(f)

    for model_param in model_config:
        if model_param["model_type"] == "DP":
            clstype = DPModel
        elif model_param["model_type"] == "ASE":
            clstype = ASEModel
        else:
            model_config_error_list.append(
                f"Model type {model_param['model_type']} is not supported."
            )
            continue
        try:
            clstype(**model_param)
        except Exception as e:
            model_config_error_list.append(f"Error in model config {model_param}: {e}")
    assert not model_config_error_list, (
        f"Model config errors: {model_config_error_list}"
    )


def test_validate_task_config():
    assert BaseTask.__subclasses__(), "No task classes found."
    for task in BaseTask.__subclasses__():
        task_config_error_list = []
        assert task.task_config.exists(), (
            f"{task.__name__} task config {task.task_config} does not exist."
        )
        with open(task.task_config, "r") as f:
            task_configs = yaml.safe_load(f)
        for task_name, task_param in task_configs.items():
            try:
                task(task_name=task_name, **task_param)
            except Exception as e:
                task_config_error_list.append(
                    f"Error in {task.task_config} task config {task_param}: {e}"
                )
    assert not task_config_error_list, f"Task config errors: {task_config_error_list}"
