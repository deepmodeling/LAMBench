import numpy as np
import pytest
from ase.data import atomic_numbers, covalent_radii

from lambench.metrics.utils import aggregated_diatomics_results
from lambench.tasks.calculator.diatomics.diatomics import (
    eval_window,
    load_reference,
    low_quality_elements,
    reference_dummy_scales,
    score_curve,
    scored_element_names,
)


def _parabola(element: str = "Si", shift: float = 0.0, scale: float = 80.0):
    r_lo, r_hi = eval_window(element, 6.0, r_min_factor=0.8)
    distances = np.linspace(r_lo + 1e-4, r_hi - 1e-4, 60)
    r0 = distances[int(len(distances) * 0.65)]
    energies = scale * (distances - (r0 + shift)) ** 2
    force_x = distances - (r0 + shift)
    return element, distances, r0, energies, force_x


def _record(**overrides):
    record = {
        "bond_length_error": 0.0,
        "wall_dist_error": 0.0,
        "force_flip_fail": 0,
        "model_excluded": False,
    }
    record.update(overrides)
    return record


def _full_results(**overrides):
    results = {name: _record() for name in scored_element_names()}
    results.update(overrides)
    return results


def test_scored_set_and_reference_gate():
    names = scored_element_names()
    assert len(names) == 87
    assert names[0] == "H"
    assert names[-1] == "U"
    for element in ("Po", "At", "Rn", "Fr", "Ra"):
        assert element not in names
    assert low_quality_elements() == frozenset(
        {"Pr", "Pm", "Sm", "Tb", "Dy", "Ho", "Er", "Tm"}
    )
    distances, energies = load_reference()["H"]
    assert distances[0] == pytest.approx(
        0.8 * float(covalent_radii[atomic_numbers["H"]])
    )
    assert distances[-1] == pytest.approx(6.0)
    assert len(distances) == len(energies)


def test_dummy_scales_are_positive():
    dummy = reference_dummy_scales()
    assert dummy["bond_length_mae"] == pytest.approx(0.6068, abs=1e-3)
    assert dummy["wall_dist_mae"] == pytest.approx(0.2733, abs=1e-3)


def test_identical_parabola_has_zero_geometry_error():
    element, distances, _r0, energies, force_x = _parabola()
    metrics = score_curve(
        element,
        distances,
        energies,
        energies,
        force_x,
        reference_low_quality=False,
    )
    assert metrics["model_excluded"] is False
    assert metrics["bond_length_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["wall_dist_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["force_flip_fail"] == 0


def test_shifted_well_moves_bond_length():
    element, distances, _r0, energies, force_x = _parabola()
    _element, _distances, _shifted_r0, shifted, shifted_fx = _parabola(shift=0.15)
    metrics = score_curve(
        element,
        distances,
        energies,
        shifted,
        shifted_fx,
        reference_low_quality=False,
    )
    assert metrics["bond_length_error"] == pytest.approx(0.15, abs=1e-6)


def test_unbound_reference_skips_bond_length():
    element, distances, r0, _energies, force_x = _parabola(scale=1e-4)
    energies = 1e-4 * (distances - r0) ** 2
    metrics = score_curve(
        element, distances, energies, energies, force_x, reference_low_quality=False
    )
    assert metrics["model_excluded"] is False
    assert metrics["bond_length_error"] is None


def test_missing_model_wall_uses_reference_radius():
    element, distances, _r0, energies, force_x = _parabola()
    flat = np.full_like(energies, energies.min())
    metrics = score_curve(
        element, distances, energies, flat, force_x, reference_low_quality=False
    )
    assert metrics["wall_dist_error"] is not None
    assert metrics["wall_dist_error"] > 0.5


def test_force_sign_must_change_once():
    element, distances, _r0, energies, force_x = _parabola()
    oscillating = np.sin(np.linspace(0, 6 * np.pi, len(distances)))
    flat = np.zeros_like(force_x)
    extra = score_curve(
        element,
        distances,
        energies,
        energies,
        oscillating,
        reference_low_quality=False,
    )
    none = score_curve(
        element, distances, energies, energies, flat, reference_low_quality=False
    )
    assert extra["force_flip_fail"] == 1
    assert none["force_flip_fail"] == 1
    assert extra["model_excluded"] is False


def test_low_quality_reference_skips_geometry():
    element, distances, _r0, energies, force_x = _parabola()
    metrics = score_curve(
        element, distances, energies, energies, force_x, reference_low_quality=True
    )
    assert metrics["model_excluded"] is False
    assert metrics["bond_length_error"] is None
    assert metrics["wall_dist_error"] is None
    assert metrics["force_flip_fail"] == 0


def test_nonfinite_and_jumpy_curves_are_excluded():
    element, distances, _r0, energies, force_x = _parabola()
    broken = energies.copy()
    broken[len(broken) // 2] = np.nan
    assert score_curve(
        element, distances, energies, broken, force_x, reference_low_quality=False
    )["model_excluded"]

    jumpy = np.zeros_like(energies)
    jumpy[10:15] += 5.0
    jumpy[25:30] -= 5.0
    jumpy[40:45] += 5.0
    assert score_curve(
        element, distances, energies, jumpy, force_x, reference_low_quality=False
    )["model_excluded"]


def test_aggregated_perfect_score_is_zero():
    agg = aggregated_diatomics_results(_full_results())
    assert agg["coverage"] == pytest.approx(1.0)
    assert agg["bond_length_mae"] == pytest.approx(0.0)
    assert agg["wall_dist_mae"] == pytest.approx(0.0)
    assert agg["force_flip_rate"] == pytest.approx(0.0)
    assert agg["score"] == pytest.approx(0.0)


def test_aggregated_caps_geometry_at_dummy():
    dummy = reference_dummy_scales()
    agg = aggregated_diatomics_results(
        _full_results(
            **{
                name: _record(bond_length_error=dummy["bond_length_mae"] * 5)
                for name in scored_element_names()
            }
        )
    )
    assert agg["score"] == pytest.approx(1.0 / 3.0)


def test_aggregated_flip_rate_and_coverage():
    names = scored_element_names()
    results = _full_results(**{name: _record(force_flip_fail=1) for name in names})
    results["H"] = _record(model_excluded=True, force_flip_fail=None)
    agg = aggregated_diatomics_results(results)
    coverage = (len(names) - 1) / len(names)
    assert agg["coverage"] == pytest.approx(coverage)
    assert agg["force_flip_rate"] == pytest.approx(1.0)
    assert agg["score"] == pytest.approx(1.0 / (3.0 * coverage))


def test_aggregated_missing_element_counts_against_coverage():
    results = _full_results()
    del results["H"]
    agg = aggregated_diatomics_results(results)
    assert agg["coverage"] == pytest.approx((len(results)) / (len(results) + 1))
    assert agg["score"] == pytest.approx(0.0)


def test_aggregated_empty_or_incomplete_terms_have_no_score():
    empty = aggregated_diatomics_results({})
    assert empty["coverage"] == pytest.approx(0.0)
    assert empty["score"] is None

    no_bond = aggregated_diatomics_results(
        _full_results(
            **{name: _record(bond_length_error=None) for name in scored_element_names()}
        )
    )
    assert no_bond["force_flip_rate"] == pytest.approx(0.0)
    assert no_bond["score"] is None
