import numpy as np
import pytest

from lambench.tasks.calculator.diatomics.diatomics import (
    _curve_metrics,
    _element_from_name,
    _scan_arrays,
)
from lambench.metrics.utils import (
    _diatomics_molecule_names,
    aggregated_diatomics_results,
)


def _full_results(**overrides: dict | None) -> dict:
    results = {name: {"roughness": 0.02} for name in _diatomics_molecule_names()}
    results.update(overrides)
    return results


def _well_scan():
    r = np.linspace(0.8, 4.0, 25)
    e = (r - 1.4) ** 2 - 1.0
    return r, e


def test_scan_arrays_drops_last_point():
    r, e = _scan_arrays(
        {"name": "SiSi", "R": [1.0, 1.2, 1.4, 1.6], "E": [0.0, -1.0, -0.5, 10.0]}
    )
    np.testing.assert_array_equal(r, [1.0, 1.2, 1.4])
    np.testing.assert_array_equal(e, [0.0, -1.0, -0.5])


def test_perfect_match_is_zero():
    r, e = _well_scan()
    metrics = _curve_metrics(r, e, e)
    assert metrics is not None
    assert metrics["roughness"] == pytest.approx(0.0)


def test_constant_offset_is_removed_by_min_shift():
    r, e = _well_scan()
    metrics = _curve_metrics(r, e + 1.5, e)
    assert metrics is not None
    assert metrics["roughness"] == pytest.approx(0.0, abs=1e-10)


def test_constant_dummy_scores_one():
    r, e = _well_scan()
    dummy = _curve_metrics(r, np.full_like(e, e[-1]), e)
    match = _curve_metrics(r, e, e)
    assert dummy is not None and match is not None
    assert dummy["roughness"] == pytest.approx(1.0)
    assert dummy["roughness"] > match["roughness"]


def test_worse_than_dummy_is_capped_at_one():
    r, e = _well_scan()
    y = e - np.min(e)
    inverted = _curve_metrics(r, np.max(y) - y, e)
    assert inverted is not None
    assert inverted["roughness"] == pytest.approx(1.0)


def test_zero_bond_step_returns_none():
    r = np.array([1.0, 1.0, 1.4])
    e = np.array([0.0, -1.0, -0.5])
    assert _curve_metrics(r, e, e) is None


def test_shape_match_is_scale_invariant():
    r, e = _well_scan()
    scaled = _curve_metrics(r, 10 * e, e)
    assert scaled is not None
    assert scaled["roughness"] == pytest.approx(0.0, abs=1e-10)


def test_oscillation_increases_roughness():
    r, e = _well_scan()
    match = _curve_metrics(r, e, e)
    wiggly = _curve_metrics(r, e + 0.2 * np.sin(25 * (r - r[0])), e)
    assert match is not None and wiggly is not None
    assert wiggly["roughness"] > match["roughness"]


def test_nonfinite_returns_none():
    r = np.array([1.0, 1.2, 1.4, 1.6])
    e = np.array([0.0, np.nan, -0.5, -0.4])
    assert _curve_metrics(r, e, e) is None


def test_element_from_name():
    assert _element_from_name("HH") == "H"
    assert _element_from_name("AlAl") == "Al"


def test_element_from_name_rejects_invalid():
    with pytest.raises(ValueError):
        _element_from_name("H2")
    with pytest.raises(ValueError):
        _element_from_name("AlH")


def test_aggregated_means():
    names = _diatomics_molecule_names()
    results = {name: {"roughness": 0.02} for name in names}
    results["HH"] = {"roughness": 0.01}
    results["NN"] = {"roughness": 0.03}
    n = len(names)
    agg = aggregated_diatomics_results(results)
    assert agg["avg_roughness"] == pytest.approx((0.01 + 0.03 + 0.02 * (n - 2)) / n)


def test_aggregated_empty_results():
    agg = aggregated_diatomics_results({})
    assert agg["avg_roughness"] is None


def test_aggregated_incomplete_coverage_is_none():
    assert aggregated_diatomics_results(_full_results(HH=None))["avg_roughness"] is None
    incomplete = _full_results()
    del incomplete["HH"]
    assert aggregated_diatomics_results(incomplete)["avg_roughness"] is None
