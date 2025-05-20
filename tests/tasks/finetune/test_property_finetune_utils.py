import json
import os
from pathlib import Path
import logging
from lambench.tasks.finetune.property_finetune import (
    PropertyFinetuneTask,
    FinetuneParams,
)
from lambench.models.dp_models import DPModel


def test_find_value_by_key_pattern_found():
    data = {"my_descriptor": {"activation_function": "relu"}, "other": 1}
    result = PropertyFinetuneTask._find_value_by_key_pattern(data, "descriptor")
    assert result == {"activation_function": "relu"}


def test_find_value_by_key_pattern_missing(caplog):
    data = {"foo": 1}
    with caplog.at_level(logging.ERROR):
        result = PropertyFinetuneTask._find_value_by_key_pattern(data, "descriptor")
    assert result is None
    assert "Descriptor not found" in caplog.text


def test_prepare_property_directory(tmp_path):
    pretrain_dir = tmp_path / "pretrain"
    pretrain_dir.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    test_dir = tmp_path / "test"
    test_dir.mkdir()

    pretrain_config = {
        "model": {
            "shared_dict": {
                "best_descriptor": {"activation_function": "gelu"},
                "type_map_all": [0, 1],
            }
        }
    }
    with open(pretrain_dir / "input.json", "w") as f:
        json.dump(pretrain_config, f)

    model = DPModel(
        model_name="m",
        model_type="DP",
        model_family="DP",
        model_path=pretrain_dir / "model.ckpt",
        virtualenv="",
        model_metadata={
            "pretty_name": "m",
            "date_added": "2025-01-01",
            "num_parameters": 1,
            "packages": {"torch": "2.0.0"},
        },
    )

    params = FinetuneParams()
    task = PropertyFinetuneTask(
        task_name="t",
        property_name="prop",
        intensive=True,
        property_dim=1,
        train_data=train_dir,
        test_data=test_dir,
        finetune_params=params,
        workdir=workdir,
    )

    os.chdir(workdir)
    task.prepare_property_directory(model)

    with open(workdir / "input.json") as f:
        cfg = json.load(f)

    assert cfg["model"]["descriptor"] == pretrain_config["model"]["shared_dict"]["best_descriptor"]
    assert "shared_dict" not in cfg["model"]
    assert cfg["training"]["training_data"]["systems"] == str(train_dir)
    assert cfg["training"]["validation_data"]["systems"] == str(test_dir)
