import numpy as np
import pytest
from ase.data import atomic_numbers, covalent_radii

from lambench.metrics.utils import aggregated_diatomics_results
from lambench.tasks.calculator.diatomics.diatomics import (
    design_distances,
    eval_window,
    load_reference,
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
        "well_depth_error": 0.0,
        "force_flip_count": 1,
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
    assert len(names) == 76
    assert names[0] == "H"
    assert names[-1] == "U"
    for element in (
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ni",
        "Zr",
        "Pr",
        "Pm",
        "Sm",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Ir",
    ):
        assert element not in names
    distances, energies = load_reference()["H"]
    assert distances[0] == pytest.approx(
        0.8 * float(covalent_radii[atomic_numbers["H"]])
    )
    assert distances[-1] == pytest.approx(6.0)
    assert len(distances) == len(energies)


def test_reference_curves_have_a_wall_and_an_interior_minimum():
    """Every stored PBE curve is a rising short-range wall on a 0.8 r_cov–6 Å grid.

    Elements that bind by at least 0.05 eV also place that minimum inside
    the scoring window rather than on an endpoint.
    """
    for element, (distances, energies) in load_reference().items():
        rcov = float(covalent_radii[atomic_numbers[element]])
        assert len(distances) == len(energies) == 50
        assert np.allclose(distances, design_distances(element), rtol=0, atol=1e-6)
        assert np.isfinite(energies).all()
        assert distances[0] == pytest.approx(0.8 * rcov)
        assert distances[-1] == pytest.approx(6.0)
        assert float(energies[0] - energies.min()) > 1.0
        r_min, r_max = eval_window(element, float(distances[-1]))
        window = (distances >= r_min) & (distances <= r_max)
        radii = distances[window]
        window_energies = energies[window]
        if float(window_energies[-1] - window_energies.min()) < 0.05:
            continue
        minimum = int(np.argmin(window_energies))
        assert 0 < minimum < len(window_energies) - 1
        assert radii[0] < radii[minimum] < radii[-1]


def test_dummy_scales_are_positive():
    dummy = reference_dummy_scales()
    assert dummy["bond_length_mae"] == pytest.approx(1.1173, abs=1e-3)
    assert dummy["well_depth_mae"] == pytest.approx(2.5920, abs=1e-3)
    assert dummy["force_flip_deviation"] == pytest.approx(1.0)


def test_identical_parabola_has_zero_reference_error():
    element, distances, _r0, energies, force_x = _parabola()
    metrics = score_curve(
        element,
        distances,
        energies,
        energies,
        force_x,
    )
    assert metrics["model_excluded"] is False
    assert metrics["bond_length_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["well_depth_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["force_flip_count"] == 1


def test_shifted_well_moves_bond_length():
    element, distances, _r0, energies, force_x = _parabola()
    _element, _distances, _shifted_r0, shifted, shifted_fx = _parabola(shift=0.15)
    metrics = score_curve(
        element,
        distances,
        energies,
        shifted,
        shifted_fx,
    )
    assert metrics["bond_length_error"] == pytest.approx(0.15, abs=1e-6)


def test_scaled_well_changes_well_depth():
    element, distances, _r0, energies, force_x = _parabola()
    scaled = energies * 0.5
    metrics = score_curve(element, distances, energies, scaled, force_x)
    r_min, r_max = eval_window(element, float(distances[-1]))
    window = (distances >= r_min) & (distances <= r_max)
    ref_depth = float(energies[window][-1] - energies[window].min())
    assert metrics["well_depth_error"] == pytest.approx(0.5 * ref_depth)


def test_unbound_reference_skips_geometry_and_well_depth():
    element, distances, r0, _energies, force_x = _parabola(scale=1e-4)
    energies = 1e-4 * (distances - r0) ** 2
    metrics = score_curve(element, distances, energies, energies, force_x)
    assert metrics["model_excluded"] is False
    assert metrics["bond_length_error"] is None
    assert metrics["well_depth_error"] is None


def test_flat_model_has_zero_depth_and_reference_depth_error():
    element, distances, _r0, energies, force_x = _parabola()
    flat = np.full_like(energies, energies.min())
    metrics = score_curve(element, distances, energies, flat, force_x)
    r_min, r_max = eval_window(element, float(distances[-1]))
    window = (distances >= r_min) & (distances <= r_max)
    ref_depth = float(energies[window][-1] - energies[window].min())
    assert metrics["well_depth_error"] == pytest.approx(ref_depth)


def test_force_flip_metric_reports_actual_count():
    element, distances, _r0, energies, force_x = _parabola()
    r_min, r_max = eval_window(element, float(distances[-1]))
    indices = np.flatnonzero((distances >= r_min) & (distances <= r_max))
    oscillating = np.ones_like(force_x)
    third = len(indices) // 3
    oscillating[indices[:third]] = -1
    oscillating[indices[third : 2 * third]] = 1
    oscillating[indices[2 * third :]] = -1
    flat = np.zeros_like(force_x)
    extra = score_curve(
        element,
        distances,
        energies,
        energies,
        oscillating,
    )
    none = score_curve(element, distances, energies, energies, flat)
    assert extra["force_flip_count"] == 2
    assert none["force_flip_count"] == 0
    assert extra["model_excluded"] is False


def test_nonfinite_curve_is_excluded_but_jumpy_curve_is_scored():
    element, distances, _r0, energies, force_x = _parabola()
    broken = energies.copy()
    broken[len(broken) // 2] = np.nan
    assert score_curve(element, distances, energies, broken, force_x)["model_excluded"]

    jumpy = np.zeros_like(energies)
    jumpy[10:15] += 5.0
    jumpy[25:30] -= 5.0
    jumpy[40:45] += 5.0
    assert not score_curve(element, distances, energies, jumpy, force_x)[
        "model_excluded"
    ]


def test_aggregated_perfect_score_is_zero():
    agg = aggregated_diatomics_results(_full_results())
    assert agg["coverage"] == pytest.approx(1.0)
    assert agg["bond_length_mae"] == pytest.approx(0.0)
    assert agg["well_depth_mae"] == pytest.approx(0.0)
    assert agg["force_flip_count"] == pytest.approx(1.0)
    assert agg["score"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("metric", "dummy_key"),
    (
        ("bond_length_error", "bond_length_mae"),
        ("well_depth_error", "well_depth_mae"),
    ),
)
def test_aggregated_caps_reference_errors_at_dummy(metric, dummy_key):
    dummy = reference_dummy_scales()
    agg = aggregated_diatomics_results(
        _full_results(
            **{
                name: _record(**{metric: dummy[dummy_key] * 5})
                for name in scored_element_names()
            }
        )
    )
    assert agg["score"] == pytest.approx(1.0 / 3.0)


def test_aggregated_flip_deviation_and_coverage():
    names = scored_element_names()
    results = _full_results(**{name: _record(force_flip_count=3) for name in names})
    results["H"] = _record(model_excluded=True, force_flip_count=None)
    agg = aggregated_diatomics_results(results)
    coverage = (len(names) - 1) / len(names)
    assert agg["coverage"] == pytest.approx(coverage)
    assert agg["force_flip_count"] == pytest.approx(3.0)
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
    assert no_bond["force_flip_count"] == pytest.approx(1.0)
    assert no_bond["score"] is None
