import logging
import tempfile
from typing import ClassVar
from pydantic import BaseModel, ConfigDict
from pathlib import Path

from lambench.databases.base_table import BaseRecord
from lambench.models.basemodel import BaseLargeAtomModel


class BaseTask(BaseModel):
    """
    BaseTask is a base class for defining and executing evaluation tasks for large atomic models.
    This class handles the task definition, execution, and result recording process. It checks if a task
    has already been run for a specific model and manages the storage of task results in a database.
    Attributes:
        task_name: A string identifying the task.
        test_data: Path to the test data for the task.
        task_config: Class variable path to the task configuration file.
        workdir: Working directory for the task, defaults to a "lambench" directory in the system's temp directory.
        record_type: Class variable defining the record type used for storing results.
        machine_type: String description of the hardware used for the task, defaults to "1 * NVIDIA V100_32g".
    Methods:
        exist(model_name): Checks if results for this task and model already exist in the database.
        run_task(model): Executes the task on the provided model if results don't already exist.
    """

    task_name: str
    test_data: Path | dict[str, Path]
    task_config: ClassVar[Path]
    model_config = ConfigDict(extra="allow")
    workdir: Path = Path(tempfile.gettempdir()) / "lambench"
    record_type: ClassVar = BaseRecord
    machine_type: str = "c12_m92_1 * NVIDIA V100"

    def exist(self, model_name: str) -> bool:
        num_records = self.record_type.count(
            task_name=self.task_name, model_name=model_name
        )
        if num_records > 1:
            logging.warning(f"Multiple records found for task {self.task_name}")
        if num_records >= 1:
            return True
        else:
            return False

    def run_task(self, model: BaseLargeAtomModel) -> None:
        if self.exist(model.model_name):
            logging.info(f"TASK {self.task_name} record found in database, SKIPPING.")
            return
        else:
            task_output = model.evaluate(task=self)
            logging.info(f"TASK {self.task_name} OUTPUT: {task_output}, INSERTING.")
            self.record_type(
                task_name=self.task_name,
                model_name=model.model_name,
                **task_output,
            ).insert()
