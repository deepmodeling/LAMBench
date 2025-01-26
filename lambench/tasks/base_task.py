import logging
import tempfile
from typing import ClassVar
from pydantic import BaseModel, ConfigDict
from pathlib import Path

from lambench.databases.base_table import BaseRecord
from lambench.models.basemodel import BaseLargeAtomModel


class BaseTask(BaseModel):
    task_name: str
    test_data: Path
    task_config: ClassVar[Path]
    model_config = ConfigDict(extra="allow")
    workdir: Path = Path(tempfile.gettempdir()) / "lambench"
    record_type: ClassVar = BaseRecord

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
