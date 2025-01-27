import logging
import os
from pathlib import Path
from typing import Optional

try:
    from deepmd.main import main as deepmd_main
except ImportError:
    logging.info("deepmd-kit is not installed, DPModel will not work.")

from lambench.models.ase_models import ASEModel
from lambench.tasks.base_task import BaseTask
from lambench.tasks import DirectPredictTask, PropertyFinetuneTask, CalculatorTask
from lambench.tasks.utils import parse_dptest_log_file


class DPModel(ASEModel):
    model_path: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "DP":
            raise ValueError(
                f"Model type {self.model_type} is not supported by DPModel"
            )

    def evaluate(self, task: BaseTask) -> Optional[dict[str, Optional[float]]]:
        if isinstance(task, DirectPredictTask | CalculatorTask):
            return super().evaluate(task)  # using ase interface
        elif isinstance(task, PropertyFinetuneTask):
            head, change_bias = None, False
        else:
            raise ValueError(f"Task {task} is not supported by DPModel")
        task.workdir.mkdir(exist_ok=True)
        os.chdir(task.workdir)

        model = self.model_path
        if isinstance(task, PropertyFinetuneTask):
            task.prepare_property_directory(self)
            model = self._finetune(model)
        elif change_bias:
            model = self._change_bias(model, task.test_data, head)
        # Optional: actually dp test can run on checkpoint
        model = self._freeze(model, head)
        test_output = self._test(model, task.test_data, head)

        if isinstance(task, PropertyFinetuneTask):
            output_type = "property"
        else:
            output_type = "standard"
        result = parse_dptest_log_file(filepath=test_output, output_type=output_type)
        return result

    @staticmethod
    def _finetune(model: Path):
        # Note: the input.json file is created under task.workdir
        command = f"dp --pt train input.json --finetune {model} --skip-neighbor-stat"
        deepmd_main(command.split()[1:])
        return Path("model.ckpt.pt")  # hard coded in deepmd-kit

    @staticmethod
    def _freeze(model: Path, head=None):
        frozen_model = Path.cwd() / model.with_suffix(".pth").name
        command = f"dp --pt freeze -c {model} -o {frozen_model} {f'--head {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        return frozen_model

    @staticmethod
    def _change_bias(model: Path, test_data: Path, head: Optional[str] = None):
        change_bias_model = Path.cwd() / f"change-bias-{model.name}"
        command = f"dp --pt change-bias {model} -o {change_bias_model} -s {test_data} {f'--model-branch {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        return change_bias_model

    @staticmethod
    def _test(model: Path, test_data: Path, head: Optional[str] = None):
        test_output = Path("dptest_output.txt")
        command = f"dp --pt test -m {model} -s {test_data} -l {test_output} {f'--head {head}' if head else ''}"
        deepmd_main(command.split()[1:])
        return test_output
