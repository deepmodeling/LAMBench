from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import itertools

from lambench.tasks.calculator.inference_efficiency.inference_efficiency import (
    run_one_inference,
    run_inference,
)


class DummyAtoms:
    def __init__(self, n):
        self.n = n

    def repeat(self, factors):
        return self

    def get_cell_lengths_and_angles(self):
        return np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])

    def copy(self):
        return self

    def __len__(self):
        return self.n

    @property
    def symbols(self):
        return "H"


def test_run_one_inference_success(monkeypatch):
    dummy_atoms = [DummyAtoms(2), DummyAtoms(2)]

    monkeypatch.setattr(
        "lambench.tasks.calculator.inference_efficiency.inference_efficiency.read",
        lambda p, s: dummy_atoms,
    )
    monkeypatch.setattr(
        "lambench.tasks.calculator.inference_efficiency.inference_efficiency.binary_search_max_natoms",
        lambda model, atoms, limit: 4,
    )
    monkeypatch.setattr(
        "lambench.tasks.calculator.inference_efficiency.inference_efficiency.find_even_factors",
        lambda n: (1, 1, 1),
    )
    monkeypatch.setattr(
        "lambench.tasks.calculator.inference_efficiency.inference_efficiency.get_efv",
        lambda atoms: None,
    )
    times = itertools.count(start=0, step=0.1)
    monkeypatch.setattr(
        "lambench.tasks.calculator.inference_efficiency.inference_efficiency.time.time",
        lambda: next(times),
    )

    model = MagicMock()
    model.calc.reset = MagicMock()

    res = run_one_inference(model, Path("dummy"), 0.0, 10)
    assert res["success_rate"] == 100.0
    assert np.isclose(res["average_time"], 50000.0)
    assert np.isclose(res["std_time"], 0.0)


def test_run_inference(monkeypatch):
    p1 = Path("a.traj")
    p2 = Path("b.traj")
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: [p1, p2])

    def mock_run_one(model, traj, w, n):
        if traj == p1:
            return {"average_time": 0.1, "std_time": 0.01, "success_rate": 100.0}
        raise RuntimeError("fail")

    monkeypatch.setattr(
        "lambench.tasks.calculator.inference_efficiency.inference_efficiency.run_one_inference",
        mock_run_one,
    )

    result = run_inference(MagicMock(), Path("test"), 0.0, 10)
    assert result[p1.name]["average_time"] == 0.1
    assert result[p1.name]["success_rate"] == 100.0
    assert result[p2.name]["average_time"] is None
    assert result[p2.name]["success_rate"] == 0.0
