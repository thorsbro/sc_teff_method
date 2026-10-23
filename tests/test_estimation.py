"""Numerical regressions use illustrative areas, never observed spectral data."""

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sc_teff_method import (
    CALIBRATION_ID,
    CONFIGURATIONS,
    LINE_IDS,
    estimate_temperature,
    get_calibration,
)

CASES = json.loads(
    (Path(__file__).parent / "fixtures/reference_regression.json").read_text()
)["cases"]
ALL_NINE = {case["configuration"]: case for case in CASES if case["id"].endswith("all_nine")}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_fixed_private_reference_regressions(case):
    result = estimate_temperature(
        case["areas"], configuration=case["configuration"], feh=case["feh"],
    )
    assert result.temperature_k == pytest.approx(case["temperature_k"], rel=0, abs=1e-9)
    assert dict(result.line_temperatures_k) == pytest.approx(
        case["line_temperatures_k"], rel=0, abs=1e-9,
    )
    if case["line_scatter_k"] is None:
        assert result.line_scatter_k is None
    else:
        assert result.line_scatter_k == pytest.approx(case["line_scatter_k"], rel=0, abs=1e-9)
    assert result.used_line_ids == tuple(case["areas"])
    assert result.n_lines == len(case["areas"])
    assert result.calibration_id == CALIBRATION_ID
    assert result.configuration == case["configuration"]
    assert result.measurement_source == "gaussian"
    assert result.temperature_scale_id == "nandakumar2023_igrins"


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("mask", range(1, 512))
def test_all_subsets_average_per_line_temperatures(configuration, mask):
    full = ALL_NINE[configuration]
    chosen = tuple(line for i, line in enumerate(LINE_IDS) if mask & (1 << i))
    # Reverse input order as an independent check of canonical line ordering.
    areas = {line: full["areas"][line] for line in reversed(chosen)}
    result = estimate_temperature(areas, configuration=configuration, feh=full["feh"])
    temperatures = [full["line_temperatures_k"][line] for line in chosen]
    expected = sum(temperatures) / len(temperatures)
    assert result.used_line_ids == chosen
    assert result.n_lines == len(chosen)
    assert result.temperature_k == pytest.approx(expected, rel=0, abs=1e-9)
    if len(temperatures) > 1:
        scatter = math.sqrt(sum((t - expected)**2 for t in temperatures) / (len(temperatures) - 1))
        assert result.line_scatter_k == pytest.approx(scatter, rel=0, abs=1e-9)
    else:
        assert result.line_scatter_k is None


def test_missing_lines_stay_absent_and_are_never_filled_with_zero():
    result = estimate_temperature({"line5": 0.8}, feh=-0.1)
    assert result.used_line_ids == ("line5",)
    assert tuple(result.line_temperatures_k) == ("line5",)
    assert result.n_lines == 1
    assert result.temperature_k == result.line_temperatures_k["line5"]
    assert result.line_scatter_k is None
    with pytest.raises(ValueError, match="positive"):
        estimate_temperature({"line5": 0.8, "line1": 0}, feh=-0.1)


def test_no_feh_is_an_independent_fit_and_optional_feh_is_diagnostic_only():
    areas = {"line1": 0.30, "line2": 0.27}
    zero_metal = estimate_temperature(areas, feh=0)
    no_metal = estimate_temperature(areas, configuration="nandakumar_no_feh")
    diagnostic = estimate_temperature(areas, configuration="nandakumar_no_feh", feh=1)
    assert zero_metal.temperature_k != pytest.approx(no_metal.temperature_k)
    assert no_metal.temperature_k == diagnostic.temperature_k
    no_feh_lines = get_calibration("nandakumar_no_feh").lines.values()
    assert all(line.b_k_per_dex is None for line in no_feh_lines)
    assert any(w.code == "outside_reference_feh_range" for w in diagnostic.warnings)
    assert not no_metal.warnings


def test_metallicity_is_applied_to_each_line_before_averaging():
    areas = {"line1": 0.30, "line2": 0.27}
    low = estimate_temperature(areas, feh=-0.2)
    high = estimate_temperature(areas, feh=0.1)
    calibration = get_calibration()
    shift = sum(calibration.lines[line].b_k_per_dex * 0.3 for line in areas) / len(areas)
    assert high.temperature_k - low.temperature_k == pytest.approx(shift, rel=0, abs=1e-9)


def test_results_and_calibration_coefficients_cannot_be_mutated():
    areas = {"line1": 0.3}
    result = estimate_temperature(areas, feh=-0.2)
    areas["line2"] = 0.27
    assert result.used_line_ids == ("line1",)
    with pytest.raises(TypeError):
        result.line_temperatures_k["line2"] = 0
    with pytest.raises(FrozenInstanceError):
        result.temperature_k = 0
    with pytest.raises(TypeError):
        get_calibration().lines["line1"] = None
    with pytest.raises(FrozenInstanceError):
        get_calibration().lines["line1"].a_k_per_angstrom = 0
    assert "correlated" in result.uncertainty_note


@pytest.mark.parametrize("areas", [{}, {"line10": 0.2}, {"Line1": 0.2}, {1: 0.2}])
def test_empty_and_unknown_line_inputs(areas):
    with pytest.raises(ValueError):
        estimate_temperature(areas, feh=-0.2)


@pytest.mark.parametrize("areas", [None, [], [("line1", 0.3)], "line1"])
def test_input_must_be_a_mapping(areas):
    with pytest.raises(TypeError, match="mapping"):
        estimate_temperature(areas, feh=-0.2)


@pytest.mark.parametrize("value", [None, "", "0.3", True, False, 0.3+0j, [], {}])
def test_area_must_be_a_real_number(value):
    with pytest.raises(TypeError, match="real number"):
        estimate_temperature({"line1": value}, feh=-0.2)


@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), -float("inf"), 10**1000])
def test_area_must_be_positive_and_finite(value):
    with pytest.raises(ValueError):
        estimate_temperature({"line1": value}, feh=-0.2)


def test_finite_input_that_overflows_prediction_is_rejected():
    with pytest.raises(ValueError, match="non-finite temperature"):
        estimate_temperature({"line1": 1e308}, feh=-0.2)


def test_feh_is_required_for_the_metallicity_dependent_model():
    with pytest.raises(ValueError, match="explicit finite feh"):
        estimate_temperature({"line1": 0.3})


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("value", ["-0.2", True, [], 0.2+0j])
def test_feh_type_validation(configuration, value):
    with pytest.raises(TypeError, match="feh"):
        estimate_temperature({"line1": 0.3}, configuration=configuration, feh=value)


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**1000])
def test_feh_finiteness_validation(configuration, value):
    with pytest.raises(ValueError, match="feh"):
        estimate_temperature({"line1": 0.3}, configuration=configuration, feh=value)


@pytest.mark.parametrize("configuration", ["apogee_mh", "", None, [], "nandakumar_FEH"])
def test_unknown_configuration_is_rejected(configuration):
    with pytest.raises(ValueError, match="configuration"):
        estimate_temperature({"line1": 0.3}, configuration=configuration, feh=-0.2)
    with pytest.raises(ValueError, match="configuration"):
        get_calibration(configuration)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_reference_temperatures(value):
    with pytest.raises(ValueError, match="reference_temperature_k"):
        estimate_temperature({"line1": 0.3}, feh=-0.2, reference_temperature_k=value)


@pytest.mark.parametrize("value", ["3500", True, [], 3500+0j])
def test_invalid_reference_temperature_types(value):
    with pytest.raises(TypeError, match="reference_temperature_k"):
        estimate_temperature({"line1": 0.3}, feh=-0.2, reference_temperature_k=value)
