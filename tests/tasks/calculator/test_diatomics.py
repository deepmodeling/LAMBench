import numpy as np
import pytest

from lambench.tasks.calculator.diatomics.diatomics import (
    _compute_roughness,
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


def test_compute_roughness_flat_residuals():
    residuals = np.zeros(10)
    assert _compute_roughness(residuals, dr=0.1) == pytest.approx(0.0)


def test_compute_roughness_oscillating():
    residuals = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
    roughness = _compute_roughness(residuals, dr=0.2)
    assert roughness is not None
    assert roughness > 0


def test_compute_roughness_dr_scaling():
    residuals = np.array([0.0, 0.1, 0.0, 0.1, 0.0], dtype=float)
    r1 = _compute_roughness(residuals, dr=0.2)
    r2 = _compute_roughness(residuals, dr=0.1)
    assert r2 == pytest.approx(4 * r1, rel=1e-6)


def test_compute_roughness_too_few_valid_points():
    residuals = np.array([np.nan, 0.5, np.nan])
    assert _compute_roughness(residuals, dr=0.1) is None


def test_compute_roughness_all_nan():
    assert _compute_roughness(np.array([np.nan, np.nan, np.nan]), dr=0.1) is None


def test_compute_roughness_matching_raw_reference_is_zero():
    residuals = np.zeros(20)
    assert _compute_roughness(residuals, dr=0.2) == pytest.approx(0.0)


def test_scan_arrays_drops_last_point():
    r, e = _scan_arrays(
        {"name": "SiSi", "R": [1.0, 1.2, 1.4, 1.6], "E": [0.0, -1.0, -0.5, 10.0]}
    )
    np.testing.assert_array_equal(r, [1.0, 1.2, 1.4])
    np.testing.assert_array_equal(e, [0.0, -1.0, -0.5])


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
    expected = (0.01 + 0.03 + 0.02 * (len(names) - 2)) / len(names)
    agg = aggregated_diatomics_results(results)
    assert agg["avg_roughness"] == pytest.approx(expected)


def test_aggregated_empty_results():
    agg = aggregated_diatomics_results({})
    assert agg["avg_roughness"] is None


def test_aggregated_incomplete_coverage_is_none():
    assert aggregated_diatomics_results(_full_results(HH=None))["avg_roughness"] is None
    assert (
        aggregated_diatomics_results(_full_results(HH={"roughness": np.nan}))[
            "avg_roughness"
        ]
        is None
    )
    incomplete = _full_results()
    del incomplete["HH"]
    assert aggregated_diatomics_results(incomplete)["avg_roughness"] is None
