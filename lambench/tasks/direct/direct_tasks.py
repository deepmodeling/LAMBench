from pathlib import Path
from typing import ClassVar, Literal
from lambench.tasks.base_task import BaseTask
from lambench.databases.direct_predict_table import DirectPredictRecord


class DirectPredictTask(BaseTask):
    """
    Support direct energy force prediction for DP interface, and zero-shot energy force prediciton for DP interface.
    For models using the ASE interface, should use `DirectPredictASETask` instead.
    """

    record_type: ClassVar = DirectPredictRecord
    task_config: ClassVar = Path(__file__).parent / "direct_tasks.yml"
    damping: Literal["d3bj", "d3zero"] | None = None
    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, test_data=kwargs["test_data"])
        self.damping = kwargs.get("damping")
