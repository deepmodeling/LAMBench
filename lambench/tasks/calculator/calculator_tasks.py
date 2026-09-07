from pathlib import Path
from typing import ClassVar

from lambench.databases.calculator_table import CalculatorRecord
from lambench.tasks.base_task import BaseTask


class CalculatorTask(BaseTask):
    """
    Support more general calculator tasks interfaced with ASE.
    """

    record_type: ClassVar = CalculatorRecord
    task_config: ClassVar = Path(__file__).parent / "calculator_tasks.yml"
    test_data: Path | None
    calculator_params: dict | None

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, **kwargs)
