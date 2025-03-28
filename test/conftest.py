from pathlib import Path
import pytest
from unittest.mock import patch

from lambench.databases.base_table import BaseRecord


def make_dummy_data():
    Path("data/dummy/test/1").mkdir(parents=True, exist_ok=True)
    Path("data/dummy/train/1").mkdir(parents=True, exist_ok=True)


# Fixtures for TASKS
@pytest.fixture
def direct_yml_data():
    make_dummy_data()
    return {
        "Example_task": {
            "test_data": Path("data/dummy/test/1"),
        }
    }


@pytest.fixture
def finetune_yml_data():
    make_dummy_data()
    return {
        "Example_task": {
            "property_name": "dipole_moment",
            "intensive": False,
            "property_dim": 1,
            "train_data": Path("data/dummy/train/1"),
            "test_data": Path("data/dummy/test/1"),
            "finetune_params": {
                "train_steps": 1000,
            },
        }
    }


@pytest.fixture
def mock_record_count():
    with patch.object(BaseRecord, "count") as mock_method:
        yield mock_method


@pytest.fixture
def mock_record_query():
    with patch.object(BaseRecord, "query") as mock_method:
        yield mock_method


@pytest.fixture
def mock_record_insert():
    with patch.object(BaseRecord, "insert") as mock_method:
        yield mock_method


@pytest.fixture
def mock_finetune_record():
    with patch("lambench.tasks.finetune.PropertyFinetuneTask.Record") as mock_method:
        yield mock_method


@pytest.fixture
def direct_task_data():
    make_dummy_data()
    return {
        "task_name": "taskA",
        "test_data": Path("data/dummy/test/1"),
    }


@pytest.fixture
def finetune_task_data():
    make_dummy_data()
    return {
        "task_name": "taskA",
        "property_name": "dipole_moment",
        "intensive": False,
        "property_dim": 1,
        "train_data": Path("data/dummy/train/1"),
        "test_data": Path("data/dummy/test/1"),
        "finetune_params": {
            "train_steps": 1000,
        },
    }


# Fixtures for MODELS
@pytest.fixture
def valid_model_data():
    return {
        "model_name": "model1",
        "model_type": "DP",
        "model_family": "DP",
        "model_path": Path("oss://lambench/DP/model.ckpt-1000.pt"),
        "virtualenv": "oss://lambench/DP/model1/venv",
        "model_metadata": {
            "pretty_name": "fancy model",
            "author": "author1",
            "description": "description1",
            "num_parameters": 1000,
            "packages": {"torch": "2.0.0"},
        },
        "show_direct_task": True,
    }


@pytest.fixture
def invalid_model_data():
    return {
        "model_name": "model1",
        "model_type": "unknown",  # invalid model_type
        "model_family": "DP",
        "model_path": Path("oss://lambench/DP/model.ckpt-1000.pt"),
        # "virtualenv": "oss://lambench/DP/model1/venv",
        "model_metadata": {
            "pretty_name": "fancy model",
            "author": "author1",
            "description": "description1",
            "num_parameters": 1000,
            "packages": {"torch": "2.0.0"},
        },
    }
