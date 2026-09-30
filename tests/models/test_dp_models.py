import pytest
from pydantic import ValidationError

from lambench.models.dp_models import DPModel


def test_load_model_valid(valid_model_data):
    model = DPModel(**valid_model_data)
    for key, value in valid_model_data.items():
        if key != "model_metadata":
            assert getattr(model, key) == value
        else:
            for k, v in value.items():
                if k == "date_added":
                    assert getattr(model.model_metadata, k).strftime("%Y-%m-%d") == v
                else:
                    assert getattr(model.model_metadata, k) == v


def test_load_model_invalid(invalid_model_data):
    with pytest.raises(ValidationError) as exc_info:
        DPModel(**invalid_model_data)
    errors = exc_info.value.errors()
    assert len(errors) == 2  # Check that two validation errors are raised.

    # Check the error for `model_type`.
    assert errors[0]["loc"] == ("model_type",)
    assert errors[0]["msg"] == "Input should be 'DP' or 'ASE'"
    assert errors[0]["type"] == "enum"

    # Check the error for missing `virtualenv`.
    assert errors[1]["loc"] == ("virtualenv",)
    assert errors[1]["msg"] == "Field required"
    assert errors[1]["type"] == "missing"
