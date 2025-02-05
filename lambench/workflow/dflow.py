import logging
import os
from pathlib import Path
from types import NoneType
from typing import Optional

from dotenv import load_dotenv

import lambench
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask
from lambench.workflow.entrypoint import job_list

load_dotenv(override=True)
# ruff: noqa: E402
from dflow import Task, Workflow
from dflow.plugins.bohrium import BohriumDatasetsArtifact, create_job_group
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow.python import OP, Artifact, PythonOPTemplate


@OP.function
def run_task_op(
    task: BaseTask,
    model: BaseLargeAtomModel,
    dataset: Artifact(Path),  # type: ignore
) -> NoneType:
    task.run_task(model)


def get_dataset(paths: list[Optional[Path]]) -> Optional[list[BohriumDatasetsArtifact]]:
    r = []
    for path in paths:
        if path is not None and str(path).startswith("/bohr/"):
            r.append(BohriumDatasetsArtifact(path))
    # due the constraint of the dflow Task, return None if no dataset, but not an empty list
    return r if r else None


def submit_tasks_dflow(
    jobs: job_list,
    name="lambench",
    image=os.environ["DEFAULT_DFLOW_IMAGE"],
    machine_type="1 * NVIDIA V100_32g",
):
    job_group_id: int = create_job_group(name)
    logging.info(
        "Job group created: "
        f"https://www.bohrium.com/jobs/list?id={job_group_id}&groupName={name}&version=v2"
    )
    wf = Workflow(name=name)
    for task, model in jobs:
        name = f"{task.task_name}--{model.model_name}"
        # dflow task name should be alphanumeric
        name = "".join([c if c.isalnum() else "-" for c in name])

        dflow_task = Task(
            name=name,
            template=PythonOPTemplate(
                run_task_op,  # type: ignore
                image=model.virtualenv if model.virtualenv else image,
                envs={k: v for k, v in os.environ.items() if k.startswith("MYSQL")},
                python_packages=[Path(package.__path__[0]) for package in [lambench]],
            ),
            parameters={
                "task": task,
                "model": model,
            },
            artifacts={"dataset": get_dataset([model.model_path, task.test_data])},
            executor=DispatcherExecutor(
                machine_dict={
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "job_name": name,
                            "bohr_job_group_id": job_group_id,
                            "platform": "ali",
                            "scass_type": machine_type,
                        },
                    },
                },
                resources_dict={
                    "source_list": [],  # for future use
                },
            ),
        )
        wf.add(dflow_task)
    wf_id = wf.submit()
    return wf_id
