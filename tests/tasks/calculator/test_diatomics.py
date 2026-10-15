import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lambench.metrics.utils import (
    _diatomic_dummy_score,
    _diatomics_molecule_names,
    aggregated_diatomics_results,
)
from lambench.tasks.calculator.diatomics.diatomics import (
    _curve_metrics,
    _element_from_name,
    _scan_arrays,
    run_inference,
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


def test_bond_position_and_depth_errors():
    r, e = _well_scan()
    exact = _curve_metrics(r, e, e)
    assert exact["bond_length_error"] == pytest.approx(0.0)
    assert exact["well_depth_error"] == pytest.approx(0.0)
    shifted = _curve_metrics(r, (r - 1.6) ** 2 - 1.0, e)
    assert shifted["bond_length_error"] == pytest.approx(0.2)
    scaled = _curve_metrics(r, e * 2, e)
    assert scaled["well_depth_error"] == pytest.approx(e[-1] - e.min())
    assert scaled["roughness"] == pytest.approx(0.0, abs=1e-10)


def test_constant_energy_dummy_scores_one_for_all_terms():
    r, e = _well_scan()
    record = _curve_metrics(r, np.zeros_like(e), e)
    agg = aggregated_diatomics_results(
        {"reference_molecules": ["HH"], "curves": {"HH": record}}
    )
    assert agg["avg_roughness"] == pytest.approx(1.0)
    assert agg["bond_length_score"] == pytest.approx(1.0)
    assert agg["well_depth_score"] == pytest.approx(1.0)
    assert agg["score"] == pytest.approx(1.0)


def test_weak_reference_still_penalizes_false_binding():
    r, e = _well_scan()
    weak = e * 0.001
    result = _curve_metrics(r, e, weak)
    assert result["bond_length_error"] is None
    assert result["well_depth_error"] > 1.0


def test_per_molecule_normalization_precedes_averaging():
    curves = {
        "HH": {
            "roughness": 0.3,
            "bond_length_error": 0.1,
            "bond_length_dummy_error": 0.2,
            "well_depth_error": 100.0,
            "well_depth_dummy_error": 1.0,
        },
        "LiLi": {
            "roughness": 0.1,
            "bond_length_error": 0.0,
            "bond_length_dummy_error": 0.1,
            "well_depth_error": 0.0,
            "well_depth_dummy_error": 1.0,
        },
    }
    result = aggregated_diatomics_results(
        {"reference_molecules": list(curves), "curves": curves}
    )
    assert result["bond_length_score"] == pytest.approx(0.25)
    assert result["well_depth_score"] == pytest.approx(0.5)
    assert result["score"] == pytest.approx((0.2 + 0.25 + 0.5) / 3)


def test_inference_uses_trimmed_custom_reference_and_roundtrips(tmp_path, monkeypatch):
    r, e = _well_scan()
    reference = [{"name": "HH", "R": r.tolist() + [5.0], "E": e.tolist() + [999.0]}]
    (tmp_path / "diatomics.json").write_text(json.dumps(reference))

    def predict(calc, element, distances):
        assert element == "H"
        np.testing.assert_array_equal(distances, r)
        return e.copy()

    monkeypatch.setattr(
        "lambench.tasks.calculator.diatomics.diatomics._predict_energies", predict
    )
    result = json.loads(json.dumps(run_inference(SimpleNamespace(calc=None), tmp_path)))
    assert result["reference_molecules"] == ["HH"]
    assert aggregated_diatomics_results(result)["score"] == pytest.approx(0.0)
    del result["curves"]["HH"]
    assert aggregated_diatomics_results(result)["score"] is None


def test_builtin_references_self_score_zero():
    path = (
        Path(__file__).resolve().parents[3]
        / "lambench/tasks/calculator/diatomics/diatomics.json"
    )
    reference = json.loads(path.read_text())
    assert len(reference) == 14
    curves = {}
    for entry in reference:
        r, e = _scan_arrays(entry)
        curves[entry["name"]] = _curve_metrics(r, e, e)
    result = aggregated_diatomics_results(
        {"reference_molecules": list(curves), "curves": curves}
    )
    assert result["score"] == pytest.approx(0.0)
    dummy_curves = {}
    for entry in reference:
        r, e = _scan_arrays(entry)
        dummy_curves[entry["name"]] = _curve_metrics(r, np.zeros_like(e), e)
    dummy = aggregated_diatomics_results(
        {"reference_molecules": list(dummy_curves), "curves": dummy_curves}
    )
    assert dummy["score"] == pytest.approx(1.0)


@pytest.mark.parametrize(("error", "expected"), [(0.0, 0.0), (0.1, 1.0)])
def test_zero_dummy_error_is_well_defined(error, expected):
    assert _diatomic_dummy_score(error, 0.0) == expected


def test_legacy_records_do_not_get_a_partial_combined_score():
    result = aggregated_diatomics_results(_full_results())
    assert result["avg_roughness"] == pytest.approx(0.02)
    assert result["score"] is None


def test_missing_bound_metric_does_not_get_a_score():
    r, e = _well_scan()
    result = _curve_metrics(r, e, e)
    result["bond_length_error"] = None
    agg = aggregated_diatomics_results(
        {"reference_molecules": ["HH"], "curves": {"HH": result}}
    )
    assert agg["score"] is None


@pytest.mark.parametrize("invalid", ["nan", "reversed", "matrix"])
def test_invalid_distance_grid_returns_none(invalid):
    r, e = _well_scan()
    if invalid == "nan":
        r[2] = np.nan
    elif invalid == "reversed":
        r = r[::-1]
    else:
        r = r[:, None]
    assert _curve_metrics(r, e, e) is None
