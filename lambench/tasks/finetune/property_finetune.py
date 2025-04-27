from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

from lambench.databases.property_table import PropertyRecord
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.tasks.base_task import BaseTask

if TYPE_CHECKING:
    from lambench.models.dp_models import DPModel


class FinetuneParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = 64
    start_lr: float = 1e-3
    stop_lr: float = 1e-4
    train_steps: int = 100000


class PropertyFinetuneTask(BaseTask):
    """
    Support property finetuning and testing for DP interface.
    Currently does not support ASE interface.
    """

    record_type: ClassVar = PropertyRecord
    task_config: ClassVar = Path(__file__).parent / "finetune_tasks.yml"
    property_name: str  # The name of the property to be finetuned, e.g. dielectric
    intensive: bool = True
    property_dim: int = 1
    train_data: Path
    finetune_params: FinetuneParams
    machine_type: str = "c48_m256_4 * NVIDIA L4"

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name=task_name, **kwargs)

    def evaluate(self, model: BaseLargeAtomModel) -> dict:
        return model.evaluate(self)

    @staticmethod
    def _find_value_by_key_pattern(d, pattern):
        for key, value in d.items():
            if pattern in key:
                return value
        else:
            logging.error("Descriptor not found in pretrain input.json!")
        return None

    def prepare_property_directory(self, model: DPModel):
        assert Path.cwd() == self.workdir, (
            f"Current working directory is {os.getcwd()}, need to change working directory to {self.workdir}!"
        )
        assert model.model_path is not None, "Model path is not specified!"
        # 1. write the finetune input.json file
        with open(os.path.join(model.model_path.parent, "input.json"), "r") as f:
            pretrain_config: dict = json.load(f)

        finetune_config = pretrain_config

        # 2. modify the input.json file
        finetune_config["learning_rate"] = {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": self.finetune_params.start_lr,
            "stop_lr": self.finetune_params.stop_lr,
            "_comment": "that's all",
        }

        finetune_config["model"]["fitting_net"] = {
            "type": "property",
            "property_name": self.property_name,
            "intensive": self.intensive,
            "task_dim": self.property_dim,
            "neuron": [240, 240, 240],
            "resnet_dt": True,
            "seed": 1,
            "_comment": " that's all",
        }
        descriptor = PropertyFinetuneTask._find_value_by_key_pattern(
            pretrain_config["model"]["shared_dict"], "descriptor"
        )
        finetune_config["model"]["descriptor"] = descriptor
        finetune_config["model"]["type_map"] = pretrain_config["model"]["shared_dict"][
            "type_map_all"
        ]
        finetune_config["model"].pop("shared_dict", None)
        finetune_config["model"].pop("model_dict", None)

        finetune_config["loss"] = {"type": "property", "_comment": " that's all"}
        finetune_config.pop("loss_dict", None)
        finetune_config["training"] = {
            "training_data": {
                "systems": str(self.train_data),
                "batch_size": f"auto:{self.finetune_params.batch_size}",
                "_comment": "that's all",
            },
            "validation_data": {
                "systems": str(self.test_data),
                "batch_size": 1,
                "_comment": "that's all",
            },
            "warmup_steps": 0,
            "gradient_max_norm": 5.0,
            "max_ckpt_keep": 10,
            "seed": 1,
            "disp_file": "lcurve.out",
            "disp_freq": self.finetune_params.train_steps // 20,
            "numb_steps": self.finetune_params.train_steps,
            "save_freq": self.finetune_params.train_steps // 5,
            "_comment": "that's all",
        }

        with open(os.path.join(self.workdir, "input.json"), "w") as f:
            json.dump(finetune_config, f, indent=4)
        logging.info(f"Finetune input file is written to {self.workdir}/input.json")
