from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict
from enum import Enum
from abc import abstractmethod
from datetime import datetime


class ModelType(str, Enum):
    DP = "DP"
    ASE = "ASE"


class SkipTaskType(str, Enum):
    DirectPredictTask = "DirectPredictTask"
    PropertyFinetuneTask = "PropertyFinetuneTask"
    CalculatorTask = "CalculatorTask"


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    pretty_name: str
    num_parameters: int
    date_added: datetime
    packages: dict[str, str]


class BaseLargeAtomModel(BaseModel):
    """
    BaseLargeAtomModel is an abstract base class for large atom models. This class defines the
    core structure and attributes that any large atom model should have. Subclasses are expected
    to implement the evaluate() method to perform model-specific evaluation on a given task.

    Attributes:
        model_name (str): The name of the model.
        model_family (str): The family or category of the model.
        model_type (ModelType): The type of the model, either `ASE` or `DP`, should use `ASE` for models using `ase` calculator interface.
        model_path (Optional[Path]): The filesystem path to the model file. Defaults to None.
        virtualenv (str): The name or path of the virtual environment required for running the model.
        model_metadata (dict[str, str]): A dictionary of metadata related to the model such as version info, author, etc.
        show_direct_task (bool): Flag indicating if the direct task should be displayed or executed. Default is True.
        show_finetune_task (bool): Flag indicating if the finetune task should be displayed or executed. Default is False.
        show_calculator_task (bool): Flag indicating if the calculator task should be displayed or executed. Default is False.
        skip_tasks (list[SkipTaskType]): List of task types that should be skipped during evaluation.
        supports_omol (bool): Flag indicating if the model is trained with OMol25 or not.
        model_domain (Optional[str]): The model head or task_name to be used for models with multiple domains. Default is None, referring to the head used for `materials` often MPTrj.
    Methods:
        evaluate(task) -> dict[str, float]:
            Abstract method for evaluating the model on a given task. Implementations should return
            a dictionary mapping evaluation metric names (as strings) to their computed scores (as floats).
    """

    model_name: str
    model_family: str
    model_type: ModelType
    model_path: Optional[Path] = None
    virtualenv: str
    model_metadata: ModelMetadata
    show_direct_task: bool = True
    show_finetune_task: bool = False
    show_calculator_task: bool = False
    skip_tasks: list[SkipTaskType] = []
    supports_omol: bool = False
    model_domain: Optional[str] = None

    @abstractmethod
    def evaluate(self, task) -> dict[str, float]:
        pass
