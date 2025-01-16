import logging
import traceback
from pathlib import Path
from typing import Type

import yaml

import lambench
from lambench.models.ase_models import ASEModel
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.models.dp_models import DPModel
from lambench.tasks import DirectPredictTask, PropertyFinetuneTask
from lambench.tasks.base_task import BaseTask

DIRECT_TASKS = Path(lambench.__file__).parent / "tasks/direct/direct_tasks.yml"
FINETUNE_TASKS = Path(lambench.__file__).parent / "tasks/finetune/finetune_tasks.yml"
MODELS = Path(lambench.__file__).parent / "models/models_config.yml"


def gather_models() -> list[BaseLargeAtomModel]:
    """
    Gather models from the models_config.yml file.
    """

    models = []
    with open(MODELS, "r") as f:
        model_config = yaml.safe_load(f)
    for model_name, model_param in model_config.items():
        if model_param["model_type"] == "DP":
            models.append(DPModel(**model_param))
        elif model_param["model_type"] == "ASE":
            models.append(ASEModel(**model_param))
        else:
            raise ValueError(f"Model type {model_param['model_type']} is not supported.")
    return models

def gather_task_type(
    models: list[BaseLargeAtomModel], task_file: Path, task_class: Type[BaseTask]
) -> list[tuple[BaseTask, BaseLargeAtomModel]]:
    """
    Gather tasks of a specific type from the task file.
    """
    tasks = []
    with open(task_file, "r") as f:
        task_configs = yaml.safe_load(f)
    for model in models:
        if isinstance(model, ASEModel) and not issubclass(task_class, DirectPredictTask):
            continue  # ASEModel only supports DirectPredictTask
        for task_name, task_param in task_configs.items():
            task = task_class(task_name=task_name, **task_param)
            if not task.exist(model.model_name):
                tasks.append((task, model))
    return tasks

def gather_jobs():
    jobs = []

    models = gather_models()
    if not models:
        logging.warning("No models found, skipping task gathering.")
        return jobs

    logging.info(f"Found {len(models)} models, gathering tasks.")
    jobs.extend(gather_task_type(models, DIRECT_TASKS, DirectPredictTask))
    # jobs.extend(gather_task_type(models, FINETUNE_TASKS, PropertyFinetuneTask)) # Not implemented yet

    return jobs

def main():
    logging.basicConfig(level=logging.INFO)

    jobs = gather_jobs()
    for task, model in jobs:
        logging.info(f"Running task={task.task_name}, model={model.model_name}")
        submit_job(task, model)

# TODO: wrap as an OP
def submit_job(task, model):
    try:
        task.run_task(model)
    except ModuleNotFoundError as e:
        logging.error(e) # Import error for ASE models
    except Exception as _:
        traceback.print_exc()
        logging.error(f"task={task.task_name}, model={model.model_name} failed!")
if __name__ == "__main__":
    main()