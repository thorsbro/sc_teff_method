"""End-to-end numerical checks use analytical synthetic spectra exclusively."""
# ruff: noqa: E402

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sc_teff_method import (
    CONFIGURATIONS,
    LINE_IDS,
    FitOptions,
    estimate_from_spectrum,
    estimate_temperature,
    fit_spectrum,
    get_calibration,
    prepare_spectrum,
    subset_performance,
)
from sc_teff_method.synthetic import make_synthetic_spectrum

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}


def analyse(case="clean", **kwargs):
    synthetic = make_synthetic_spectrum(case)
    return fit_spectrum(
        synthetic.wavelength_air_angstrom, synthetic.flux,
        uncertainty=synthetic.uncertainty, mask=synthetic.mask, rv_kms=synthetic.rv_kms,
        **CONVENTIONS, **kwargs,
    )


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("scope", ["strict_strength_domain", "all_available"])
def test_independent_analytic_areas_and_pr1_temperature_agree(configuration, scope):
    synthetic = make_synthetic_spectrum()
    result = analyse(configuration=configuration, feh=-0.2, validation_scope=scope)
    truth = {line: math.sqrt(2*math.pi) * p.amplitude * p.sigma_angstrom
             for line, p in synthetic.truth.items()}
    expected = estimate_temperature(truth, configuration=configuration, feh=-0.2)
    assert result.included_line_ids == LINE_IDS
    assert result.temperature.temperature_k == pytest.approx(expected.temperature_k, abs=1e-7)
    assert dict(result.temperature.line_temperatures_k) == pytest.approx(
        dict(expected.line_temperatures_k), abs=1e-7,
    )
    assert result.temperature.line_scatter_k == pytest.approx(expected.line_scatter_k, abs=1e-7)
    assert result.performance == subset_performance(
        LINE_IDS, configuration=configuration, scope=scope,
    )
    assert result.performance.cohort == "exact_subset"
    assert all(fit.requires_inspection and fit.window_provisional for fit in result.fits.values())
    assert not any(fit.flags for fit in result.fits.values())


def test_subset_exclusions_and_missing_coverage_never_become_zero():
    result = analyse("missing", feh=-0.2, excluded_lines=["line8"])
    assert result.included_line_ids == ("line1", "line2", "line4", "line5", "line6", "line9")
    assert result.performance.n_stars == 44
    for line, status in (("line3", "unavailable"), ("line7", "unavailable"), ("line8", "excluded")):
        assert result.fits[line].status == status
        assert result.fits[line].area_angstrom is None
        assert line not in result.line_temperatures_k
        assert line in result.exclusion_reasons
    expected = estimate_temperature(
        {line: result.fits[line].area_angstrom for line in result.included_line_ids}, feh=-0.2,
    )
    assert result.temperature == expected


def test_converged_blended_lines_require_explicit_inclusion():
    strict = analyse("blends", feh=-0.2)
    deliberate = analyse("blends", feh=-0.2, quality_policy="include_warned")
    for line in ("line4", "line6", "line8"):
        assert strict.fits[line].status == "converged"
        assert strict.fits[line].requires_inspection
        assert strict.fits[line].flags
        assert line not in strict.included_line_ids
        assert line in strict.line_temperatures_k  # diagnostic only
        assert line in deliberate.included_line_ids
    assert deliberate.included_line_ids == LINE_IDS
    assert any("explicit include_warned" in w for w in deliberate.warnings)
    assert deliberate.temperature.temperature_k != pytest.approx(strict.temperature.temperature_k)


def test_poor_sampling_is_marked_and_excluded_by_default():
    result = analyse("variable", feh=-0.2)
    for line in ("line8", "line9"):
        assert "poor_fwhm_sampling" in result.fits[line].flags
        assert line not in result.included_line_ids
    assert result.temperature.n_lines == 7


def test_absolute_pixel_uncertainties_produce_formal_errors_without_teff_error_claim():
    synthetic = make_synthetic_spectrum("noise")
    result = analyse("noise", feh=-0.2)
    assert result.temperature.n_lines == 9
    for line, truth in synthetic.truth.items():
        fit = result.fits[line]
        assert fit.weighting == "inverse_sigma"
        assert fit.area_error_angstrom > 0
        assert abs(fit.area_angstrom-truth.area_angstrom) < 5 * fit.area_error_angstrom
    assert any("unweighted measurements" in w for w in result.warnings)
    assert "not the total temperature uncertainty" in result.temperature.uncertainty_note


@pytest.mark.parametrize("rv", [-65.0, 0.0, 65.0])
def test_velocity_sign_and_width_scaling_are_consistent(rv):
    synthetic = make_synthetic_spectrum()
    beta = rv / 299792.458
    factor = math.sqrt((1+beta)/(1-beta))
    result = fit_spectrum(
        synthetic.wavelength_air_angstrom*factor, synthetic.flux,
        rv_kms=rv, feh=-0.2, **CONVENTIONS,
    )
    for line, truth in synthetic.truth.items():
        assert result.fits[line].parameters.centre_air_angstrom == pytest.approx(
            truth.centre_air_angstrom, abs=1e-7,
        )
        assert result.fits[line].area_angstrom == pytest.approx(truth.area_angstrom, abs=1e-8)
    assert result.spectrum.rv_kms == rv


def test_rv_scenario_is_not_corrected_unless_explicitly_requested():
    synthetic = make_synthetic_spectrum("rv_offset")
    result = fit_spectrum(synthetic.wavelength_air_angstrom, synthetic.flux,
                          feh=-0.2, **CONVENTIONS)
    assert result.spectrum.rv_kms is None
    assert any(fit.flags for fit in result.fits.values())
    assert len(result.included_line_ids) < 9


@pytest.mark.parametrize("policy", ["exclude_warned", "include_warned"])
def test_failed_fits_never_contribute_even_with_include_warned(policy):
    result = analyse(feh=-0.2, options=FitOptions(max_nfev=1), quality_policy=policy)
    assert result.temperature is None
    assert result.performance is None
    assert not result.included_line_ids
    assert not result.line_temperatures_k
    assert all(fit.status == "failed" and fit.area_angstrom is None for fit in result.fits.values())


def test_calibration_extrapolation_and_strict_scope_caveat_survive_fitting():
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    x = rest + np.linspace(-4, 4, 801)
    y = 1 - 0.03*np.exp(-0.5*((x-rest)/0.1)**2)
    result = fit_spectrum(x, y, feh=-0.2, **CONVENTIONS)
    assert result.included_line_ids == ("line1",)
    assert result.target_passes_declared_strength_cut is False
    assert any("strict-domain" in w for w in result.warnings)
    assert {w.code for w in result.temperature.warnings} >= {
        "outside_training_strength_range", "outside_declared_strength_cut",
    }
    assert result.temperature == estimate_temperature(
        {"line1": result.fits["line1"].area_angstrom}, feh=-0.2,
    )


def test_every_line_can_be_explicitly_excluded_and_stays_missing():
    result = analyse(feh=-0.2, excluded_lines=LINE_IDS)
    assert result.temperature is None and result.performance is None
    assert all(fit.status == "excluded" for fit in result.fits.values())
    assert not result.line_temperatures_k


@pytest.mark.parametrize("kwargs,error", [
    ({}, ValueError), ({"feh": float("nan")}, ValueError), ({"feh": True}, TypeError),
    ({"configuration": "unknown", "feh": 0}, ValueError),
    ({"feh": 0, "quality_policy": "best"}, ValueError),
    ({"feh": 0, "validation_scope": "training_range"}, ValueError),
    ({"feh": 0, "reference_temperature_k": -1}, ValueError),
])
def test_workflow_input_validation(kwargs, error):
    with pytest.raises(error):
        analyse(**kwargs)


def test_prepared_spectrum_and_array_interfaces_agree_without_double_rv():
    synthetic = make_synthetic_spectrum("rv_offset")
    spectrum = prepare_spectrum(synthetic.wavelength_air_angstrom, synthetic.flux,
                                rv_kms=synthetic.rv_kms, **CONVENTIONS)
    prepared = estimate_from_spectrum(spectrum, feh=-0.2)
    direct = analyse("rv_offset", feh=-0.2)
    assert prepared.temperature == direct.temperature
    assert prepared.spectrum.doppler_factor == direct.spectrum.doppler_factor
