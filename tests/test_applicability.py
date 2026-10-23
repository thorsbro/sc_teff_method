"""Declared selection and empirical marginal coverage remain distinct."""

import math

import pytest

from sc_teff_method import CONFIGURATIONS, LINE_IDS, estimate_temperature, get_calibration


def codes(result):
    return {warning.code for warning in result.warnings}


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_declared_cut_is_inclusive_and_does_not_remove_inputs(configuration):
    at_cut = estimate_temperature({"line1": 0.08}, configuration=configuration, feh=-0.2)
    below = estimate_temperature(
        {"line1": math.nextafter(0.08, 0)}, configuration=configuration, feh=-0.2,
    )
    assert "outside_declared_strength_cut" not in codes(at_cut)
    assert "outside_training_strength_range" in codes(at_cut)
    assert "outside_declared_strength_cut" in codes(below)
    assert "outside_training_strength_range" in codes(below)
    assert below.used_line_ids == ("line1",)
    assert below.n_lines == 1
    assert all(w.line_id == "line1" for w in below.warnings)
    assert any("strict_strength_domain" in w.message for w in below.warnings)


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("line_id", LINE_IDS)
@pytest.mark.parametrize("boundary", ["minimum", "maximum"])
def test_empirical_strength_boundaries_are_inclusive(configuration, line_id, boundary):
    limits = get_calibration(configuration).lines[line_id].training_strength_range_angstrom
    edge = limits[0 if boundary == "minimum" else 1]
    outside = math.nextafter(edge, 0 if boundary == "minimum" else math.inf)
    inside_result = estimate_temperature({line_id: edge}, configuration=configuration, feh=-0.2)
    outside_result = estimate_temperature({line_id: outside}, configuration=configuration, feh=-0.2)
    assert "outside_training_strength_range" not in codes(inside_result)
    assert "outside_training_strength_range" in codes(outside_result)
    assert "outside_declared_strength_cut" not in codes(outside_result)
    assert outside_result.n_lines == 1


def test_declared_selection_has_no_upper_strength_cut():
    calibration = get_calibration()
    assert calibration.declared_strength_min_angstrom == 0.08
    assert calibration.declared_strength_max_angstrom is None
    result = estimate_temperature({"line1": 2}, feh=-0.2)
    assert "outside_training_strength_range" in codes(result)
    assert "outside_declared_strength_cut" not in codes(result)
    assert result.n_lines == 1


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_line3_has_narrower_reference_coverage(configuration):
    result = estimate_temperature(
        {"line1": 0.3, "line3": 0.3}, configuration=configuration,
        feh=-0.8, reference_temperature_k=3750,
    )
    assert {(w.line_id, w.code) for w in result.warnings} == {
        ("line3", "outside_reference_temperature_range"),
        ("line3", "outside_reference_feh_range"),
    }
    assert result.used_line_ids == ("line1", "line3")
    assert result.n_lines == 2


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("line_id", LINE_IDS)
def test_metallicity_and_temperature_reference_boundaries(configuration, line_id):
    line = get_calibration(configuration).lines[line_id]
    for metal in line.reference_feh_range_dex:
        for temperature in line.reference_temperature_range_k:
            area = sum(line.training_strength_range_angstrom) / 2
            result = estimate_temperature(
                {line_id: area}, configuration=configuration,
                feh=metal, reference_temperature_k=temperature,
            )
            assert not result.warnings
    for metal in (line.reference_feh_range_dex[0] - 0.01, line.reference_feh_range_dex[1] + 0.01):
        result = estimate_temperature({line_id: area}, configuration=configuration, feh=metal)
        assert "outside_reference_feh_range" in codes(result)
    for temperature in (
        line.reference_temperature_range_k[0] - 1, line.reference_temperature_range_k[1] + 1,
    ):
        result = estimate_temperature(
            {line_id: area}, configuration=configuration,
            feh=-0.2, reference_temperature_k=temperature,
        )
        assert "outside_reference_temperature_range" in codes(result)


def test_predicted_temperature_is_only_a_proxy_and_external_reference_overrides_it():
    inferred = estimate_temperature({"line1": 0.08}, feh=0.25)
    external = estimate_temperature({"line1": 0.08}, feh=0.25, reference_temperature_k=3500)
    assert "outside_reference_temperature_range" in codes(inferred)
    assert any("applicability proxy" in w.message for w in inferred.warnings)
    assert "outside_reference_temperature_range" not in codes(external)
    assert external.temperature_k == inferred.temperature_k
    assert external.used_line_ids == inferred.used_line_ids
