import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, Type, TypeAlias

import yaml

import lambench
from lambench.models.ase_models import ASEModel
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.models.dp_models import DPModel
from lambench.tasks import PropertyFinetuneTask
from lambench.tasks.base_task import BaseTask

MODELS = Path(lambench.__file__).parent / "models/models_config.yml"


def gather_model_params(
    model_names: Optional[list[str]] = None,
) -> list[dict]:
    """
    Gather model parameters from the models_config.yml file for selected models.
    """

    model_params = []
    with open(MODELS, "r") as f:
        model_config: list[dict] = yaml.safe_load(f)
    for model_param in model_config:
        if model_names and model_param["model_name"] not in model_names:
            continue
        model_params.append(model_param)

    return model_params


def gather_model(model_param: dict, model_domain: str) -> BaseLargeAtomModel:
    model_param = model_param.copy()
    model_param["model_domain"] = model_domain
    if model_param["model_type"] == "DP":
        return DPModel(**model_param)
    elif model_param["model_type"] == "ASE":
        return ASEModel(**model_param)
    else:
        raise ValueError(f"Model type {model_param['model_type']} is not supported.")


job_list: TypeAlias = list[tuple[BaseTask, BaseLargeAtomModel]]


def gather_task_type(
    model_params: list[dict],
    task_class: Type[BaseTask],
    task_names: Optional[list[str]] = None,
) -> job_list:
    """
    Gather tasks of a specific type from the task file.
    """
    tasks = []
    with open(task_class.task_config, "r") as f:
        task_configs: dict[str, dict] = yaml.safe_load(f)
    for model_param in model_params:
        if model_param["model_type"] != "DP" and issubclass(
            task_class, PropertyFinetuneTask
        ):
            continue  # Regular ASEModel does not support PropertyFinetuneTask
        for task_name, task_params in task_configs.items():
            if (task_names and task_name not in task_names) or task_class.__name__ in (
                model_param.get("skip_tasks", [])
            ):
                continue
            task = task_class(task_name=task_name, **task_params)
            if not task.exist(model_param["model_name"]):
                # model_domain = task.domain if task.domain else "" # in the future we may have tasks with specific domain.

                # currently only need to distinguish direct tasks for molecules and materials due to OMol25 training set.
                if task_name in ["AQM", "H_nature_2022", "AIMD_Chig"]:
                    model_domain = "molecules"
                else:
                    model_domain = "materials"
                model = gather_model(model_param, model_domain)
                tasks.append((task, model))
    return tasks


def gather_jobs(
    model_names: Optional[list[str]] = None,
    task_names: Optional[list[str]] = None,
    task_types: Optional[list[Type[BaseTask]]] = None,
) -> job_list:
    jobs: job_list = []

    model_params = gather_model_params(model_names)
    if not model_params:
        logging.warning("No models found, skipping task gathering.")
        return jobs

    logging.info(f"Found {len(model_params)} models, gathering tasks.")
    for task_class in BaseTask.__subclasses__():
        if task_types and task_class.__name__ not in task_types:
            continue
        jobs.extend(
            gather_task_type(
                model_params=model_params, task_class=task_class, task_names=task_names
            )
        )

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run tasks for models.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="The model names in `models_config.yml`. e.g. --models DP_2024Q4 MACE_MP_0 SEVENNET_0",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        help="The task names in `direct_tasks.yml` or `finetune_tasks.yml`. e.g. --tasks HPt_NC_2022 Si_ZEO22",
    )
    parser.add_argument(
        "--task-types",
        type=str,
        nargs="*",
        help="The task types. e.g. --task-types PropertyFinetuneTask",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run tasks locally.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    jobs = gather_jobs(
        model_names=args.models, task_names=args.tasks, task_types=args.task_types
    )
    if not jobs:
        logging.warning("No jobs found, exiting.")
        return
    logging.info(f"Found {len(jobs)} jobs.")
    if args.local:
        submit_tasks_local(jobs)
    else:
        from lambench.workflow.dflow import submit_tasks_dflow

        submit_tasks_dflow(jobs)


def submit_tasks_local(jobs: job_list) -> None:
    for task, model in jobs:
        logging.info(f"Running task={task.task_name}, model={model.model_name}")
        try:
            task.run_task(model)
        except ModuleNotFoundError as e:
            logging.error(e)  # Import error for ASE models
        except Exception as _:
            traceback.print_exc()
            logging.error(f"task={task.task_name}, model={model.model_name} failed!")


if __name__ == "__main__":
    main()
