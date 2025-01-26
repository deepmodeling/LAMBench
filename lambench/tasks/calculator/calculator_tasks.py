from typing import ClassVar, Optional
from pathlib import Path
from lambench.tasks.base_task import BaseTask
from lambench.databases.calculator_table import CalculatorRecord


class CalculatorTask(BaseTask):
    """
    Support more general calculator tasks interfaced with ASE.
    """

    record_type: ClassVar = CalculatorRecord
    task_config: ClassVar = Path(__file__).parent / "calculator_tasks.yml"
    test_data: Optional[Path]
    calculator_params: dict

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, **kwargs)
