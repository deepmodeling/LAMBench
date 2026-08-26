import logging
import os
from pathlib import Path
from types import NoneType
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)
# ruff: noqa: E402
from dflow import Task, Workflow
from dflow.plugins.bohrium import BohriumDatasetsArtifact, create_job_group
from dflow.plugins.dispatcher import DispatcherExecutor
from dflow.python import OP, Artifact, PythonOPTemplate

import dpdata

import lambench
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask

# ASEModel imports dftd3 at module load, so the worker must install it before
# `from lambench.workflow.dflow import run_task_op`. PythonOPTemplate prepends
# this block to `script`. Do not use str.format placeholders here: dflow calls
# pre_script.format(tmp_root=...). Worker images often pin a Tsinghua PyPI
# mirror that 403s on dftd3 wheels; force indexes that Bohrium can reach.
_DFTD3_PRE_SCRIPT = """\
import importlib.util
import os
import subprocess
import sys

os.environ.pop("PIP_INDEX_URL", None)
os.environ.pop("PIP_EXTRA_INDEX_URL", None)
if importlib.util.find_spec("dftd3") is None:
    install = [sys.executable, "-m", "pip", "install", "dftd3"]
    try:
        subprocess.check_call(install + ["--index-url", "https://mirrors.aliyun.com/pypi/simple", "--trusted-host", "mirrors.aliyun.com"])
    except subprocess.CalledProcessError:
        subprocess.check_call(install + ["--index-url", "https://pypi.org/simple", "--trusted-host", "pypi.org"])
"""


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
    jobs: list[tuple[BaseTask, BaseLargeAtomModel]],
    name="lambench",
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
        if task.test_data is not None:
            # handle dict type test_data, NOTE: if the datasets are in the same parent folder, only need to upload the artifact once.
            task_data = (
                list(task.test_data.values())[0]
                if isinstance(task.test_data, dict)
                else task.test_data
            )
        else:
            task_data = []
        logging.warning(f"Submitting task {name} with test data paths: {task_data}")

        dflow_task = Task(
            name=name,
            template=PythonOPTemplate(
                run_task_op,  # type: ignore
                image=model.virtualenv,
                envs={k: v for k, v in os.environ.items() if k.startswith("MYSQL")},
                python_packages=[
                    Path(package.__path__[0]) for package in [lambench, dpdata]
                ],
                pre_script=_DFTD3_PRE_SCRIPT,
            ),
            parameters={
                "task": task,
                "model": model,
            },
            artifacts={"dataset": get_dataset([model.model_path, task_data])},
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
                            "scass_type": task.machine_type,
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
