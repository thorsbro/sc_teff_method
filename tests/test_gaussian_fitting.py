"""Numerical tests use only analytic Gaussian profiles and seeded artificial noise."""
# ruff: noqa: E402

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sc_teff_method import LINE_IDS, get_calibration
from sc_teff_method import fitting as module
from sc_teff_method.fitting import (
    FitOptions,
    GaussianParameters,
    fit_gaussian_lines,
    fit_pixel_masks,
    gaussian_absorption,
    gaussian_area,
)
from sc_teff_method.spectrum import prepare_spectrum

REST = get_calibration().lines["line1"].wavelength_air_angstrom
PARAMETERS = GaussianParameters(0.32, REST+0.07, 0.35)


def analytic(x, amplitude=0.32, centre=REST+0.07, sigma=0.35):
    """Independent analytical formula; no fitted or observed template is used."""
    return 1-amplitude*np.exp(-(x-centre)**2/(2*sigma**2))


def prepared(*, step=0.025, amplitude=0.32, centre=REST+0.07, sigma=0.35,
             noise=0, uncertainty=None, mask=None):
    x = REST+np.arange(-5, 5+step/2, step)
    flux = analytic(x, amplitude, centre, sigma)
    if noise:
        flux += np.random.default_rng(8675309).normal(0, noise, len(x))
    if uncertainty is not None:
        uncertainty = np.full(len(x), uncertainty)
    return prepare_spectrum(
        x, flux, wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
        uncertainty=uncertainty, mask=mask,
    )


def test_gaussian_model_and_area_match_independent_analytic_integral():
    offsets = np.linspace(-5, 5, 40001)
    x = REST+offsets
    profile = gaussian_absorption(x, PARAMETERS)
    assert profile == pytest.approx(analytic(x), rel=0, abs=2e-16)
    # Uniform-grid quadrature over artificial analytic samples independently
    # checks the infinite-profile normalization to negligible Gaussian tails.
    step = offsets[1]-offsets[0]
    integral = float(np.sum(1-profile)*step)
    expected = math.sqrt(2*math.pi)*0.32*0.35
    assert integral == pytest.approx(expected, rel=0, abs=1e-12)
    assert gaussian_area(0.32, 0.35) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize("amplitude,sigma,offset", [
    (0.12, 0.12, -0.2), (0.32, 0.35, 0.07), (0.6, 0.8, 0.1), (0.3, 1.1, -0.05),
])
def test_noiseless_recovery_of_centres_depths_widths_and_areas(amplitude, sigma, offset):
    spectrum = prepared(amplitude=amplitude, sigma=sigma, centre=REST+offset)
    fit = fit_gaussian_lines(spectrum)["line1"]
    assert fit.status == "converged"
    assert fit.parameters.amplitude == pytest.approx(amplitude, rel=0, abs=1e-8)
    assert fit.parameters.centre_air_angstrom == pytest.approx(REST+offset, rel=0, abs=1e-8)
    assert fit.parameters.sigma_angstrom == pytest.approx(sigma, rel=0, abs=1e-8)
    assert fit.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*amplitude*sigma, abs=1e-8)
    assert fit.residual_rms < 1e-9
    assert fit.requires_inspection
    assert fit.flags == ()


def test_seeded_noise_recovery_is_consistent_with_known_noise_level():
    fit = fit_gaussian_lines(prepared(noise=0.003, uncertainty=0.003))["line1"]
    assert fit.status == "converged"
    assert fit.parameters.amplitude == pytest.approx(0.32, abs=0.004)
    assert fit.parameters.centre_air_angstrom == pytest.approx(REST+0.07, abs=0.006)
    assert fit.parameters.sigma_angstrom == pytest.approx(0.35, abs=0.006)
    assert fit.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*0.32*0.35, abs=0.004)
    assert fit.residual_rms == pytest.approx(0.003, abs=0.0007)
    assert 0.5 < fit.reduced_chi2 < 1.7
    assert fit.area_error_angstrom > 0
    assert fit.weighting == "inverse_sigma"


def test_known_absolute_uncertainties_are_not_rescaled_to_zero_residuals():
    first = fit_gaussian_lines(prepared(uncertainty=0.003))["line1"]
    doubled = fit_gaussian_lines(prepared(uncertainty=0.006))["line1"]
    assert first.residual_rms < 1e-10
    assert first.area_error_angstrom > 0
    assert doubled.area_error_angstrom == pytest.approx(2*first.area_error_angstrom, rel=1e-8)
    assert np.asarray(doubled.covariance) == pytest.approx(4*np.asarray(first.covariance), rel=1e-8)


@pytest.mark.parametrize("known_uncertainty", [None, 0.003])
def test_formal_area_error_includes_amplitude_sigma_covariance(known_uncertainty):
    spectrum = prepared(noise=0.003, uncertainty=known_uncertainty)
    fit = fit_gaussian_lines(spectrum)["line1"]
    _, used = fit_pixel_masks(spectrum, fit.window)
    x = spectrum.wavelength_air_angstrom[used]
    amplitude, centre, sigma = (
        fit.parameters.amplitude, fit.parameters.centre_air_angstrom, fit.parameters.sigma_angstrom,
    )
    difference = x-centre
    exponential = np.exp(-difference**2/(2*sigma**2))
    # Independently calculate the normal-equation covariance and full area
    # derivative, including the off-diagonal amplitude-width contribution.
    jacobian = np.column_stack((
        -exponential, -amplitude*exponential*difference/sigma**2,
        -amplitude*exponential*difference**2/sigma**3,
    ))
    if known_uncertainty is not None:
        jacobian /= known_uncertainty
    covariance = np.linalg.inv(jacobian.T@jacobian)
    if known_uncertainty is None:
        residuals = analytic(x, amplitude, centre, sigma)-spectrum.flux[used]
        covariance *= np.dot(residuals, residuals)/(len(x)-3)
        assert fit.reduced_chi2 is None
    gradient = math.sqrt(2*math.pi)*np.array([sigma, 0, amplitude])
    expected_error = math.sqrt(float(gradient@covariance@gradient))
    assert np.asarray(fit.covariance) == pytest.approx(covariance, rel=2e-8, abs=1e-15)
    assert fit.area_error_angstrom == pytest.approx(expected_error, rel=2e-8)
    diagonal_only_error = math.sqrt(float(gradient@np.diag(np.diag(covariance))@gradient))
    assert fit.area_error_angstrom < diagonal_only_error
    assert fit.parameter_errors.amplitude == pytest.approx(math.sqrt(covariance[0, 0]))


def test_all_nine_lines_are_retained_when_only_one_has_coverage():
    fits = fit_gaussian_lines(prepared())
    assert tuple(fits) == LINE_IDS
    for line in LINE_IDS[1:]:
        assert fits[line].status == "unavailable"
        assert fits[line].area_angstrom is None
        assert fits[line].parameters is None
        assert fits[line].flags == ("no_coverage",)
    with pytest.raises(TypeError):
        fits["line1"] = None
    with pytest.raises(FrozenInstanceError):
        fits["line1"].area_angstrom = 0


def test_window_initial_guess_and_explicit_line_exclusion_overrides():
    initial = GaussianParameters(0.4, REST+0.15, 0.5)
    fits = fit_gaussian_lines(
        prepared(), windows={"line1": (REST-2, REST+2)},
        initial_parameters={"line1": initial}, excluded_lines=["line2"],
    )
    assert fits["line1"].window == (REST-2, REST+2)
    assert not fits["line1"].window_provisional
    assert fits["line1"].area_angstrom == pytest.approx(math.sqrt(2*math.pi)*0.32*0.35, abs=1e-9)
    assert fits["line2"].status == "excluded"
    assert fits["line2"].area_angstrom is None
    assert fits["line3"].window_provisional


def test_fully_masked_window_stays_unavailable_with_no_area():
    spectrum = prepared(mask=np.ones(401, dtype=bool))
    fit = fit_gaussian_lines(spectrum)["line1"]
    assert fit.status == "unavailable"
    assert fit.n_pixels == 0
    assert fit.n_excluded > 200
    assert fit.area_angstrom is None
    assert "insufficient_samples" in fit.flags


def test_masks_and_exclusion_intervals_select_exactly_the_used_pixels():
    spectrum = prepared()
    window = (REST-3, REST+3)
    excluded = ((REST+0.8, REST+1.1),)
    region, used = fit_pixel_masks(spectrum, window, excluded)
    x = spectrum.wavelength_air_angstrom
    assert np.array_equal(region, (x >= window[0]) & (x <= window[1]))
    assert not np.any(used & (x >= REST+0.8) & (x <= REST+1.1))
    fit = fit_gaussian_lines(spectrum, excluded_regions={"line1": excluded})["line1"]
    assert fit.n_pixels == int(np.sum(used))
    assert fit.n_excluded == int(np.sum(region & ~used))
    assert fit.excluded_regions == excluded
    assert "masked_gap" in fit.flags
    assert fit.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*0.32*0.35, abs=1e-8)


def test_blend_warning_and_explicit_contamination_mask():
    x = REST+np.arange(-5, 5.01, 0.025)
    flux = analytic(x)-0.25*np.exp(-(x-(REST+1))**2/(2*0.12**2))
    spectrum = prepare_spectrum(
        x, flux, wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
    )
    blended = fit_gaussian_lines(spectrum)["line1"]
    masked = fit_gaussian_lines(
        spectrum, excluded_regions={"line1": ((REST+0.5, REST+1.5),)},
    )["line1"]
    assert "high_residual_rms" in blended.flags
    assert "high_residual_rms" not in masked.flags
    assert "masked_gap" in masked.flags
    assert masked.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*0.32*0.35, abs=0.001)


def test_missing_internal_pixels_and_line_core_gaps_are_detected():
    x = REST+np.arange(-5, 5.01, 0.025)
    retain = (x < REST-0.4) | (x > REST+0.4)
    spectrum = prepare_spectrum(
        x[retain], analytic(x[retain]), wavelength_unit="angstrom", wavelength_frame="air",
        continuum_normalised=True,
    )
    fit = fit_gaussian_lines(spectrum)["line1"]
    assert "missing_or_irregular_pixels" in fit.flags
    assert "internal_gap" in fit.flags
    assert "line_core_gap" in fit.flags


def test_partial_window_coverage_is_a_quality_warning():
    x = REST+np.arange(-0.7, 3.1, 0.025)
    spectrum = prepare_spectrum(
        x, analytic(x), wavelength_unit="angstrom", wavelength_frame="air",
        continuum_normalised=True,
    )
    fit = fit_gaussian_lines(spectrum)["line1"]
    assert fit.status == "converged"
    assert "partial_window_coverage" in fit.flags


def test_poor_sampling_is_reported_even_if_an_optimizer_converges():
    fit = fit_gaussian_lines(prepared(step=0.15, sigma=0.14))["line1"]
    assert fit.status == "converged"
    assert "poor_fwhm_sampling" in fit.flags


@pytest.mark.parametrize("change,expected_flag", [
    ({"amplitude": 1.2}, "amplitude_near_bound"),
    ({"sigma": 0.012}, "sigma_near_bound"),
    ({"centre": REST+1.4}, "centre_near_bound"),
])
def test_parameter_boundary_flags_suppress_formal_errors(change, expected_flag):
    fit = fit_gaussian_lines(prepared(**change))["line1"]
    assert fit.status == "converged"
    assert expected_flag in fit.flags
    assert "covariance_unavailable_at_boundary" in fit.flags
    assert fit.area_error_angstrom is None
    assert fit.parameter_errors is None
    assert fit.covariance is None


def test_large_centre_shift_is_reported_separately_from_convergence():
    fit = fit_gaussian_lines(prepared(centre=REST+0.8))["line1"]
    assert fit.status == "converged"
    assert fit.centre_displacement_angstrom == pytest.approx(0.8, abs=1e-8)
    assert "large_centre_displacement" in fit.flags


def test_supplied_underestimated_noise_flags_high_reduced_chi_square():
    fit = fit_gaussian_lines(prepared(noise=0.005, uncertainty=0.001))["line1"]
    assert fit.status == "converged"
    assert fit.reduced_chi2 > 3
    assert "high_reduced_chi2" in fit.flags


def test_optimizer_failure_preserves_missingness():
    fit = fit_gaussian_lines(
        prepared(), options=FitOptions(max_nfev=1),
        initial_parameters={"line1": GaussianParameters(0.9, REST+0.9, 1.3)},
    )["line1"]
    assert fit.status == "failed"
    assert fit.nfev == 1
    assert "optimizer_failed" in fit.flags
    assert fit.area_angstrom is None
    assert fit.parameters is None


def test_optimizer_exception_becomes_explicit_failed_line(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate synthetic optimizer failure")
    monkeypatch.setattr(module, "least_squares", fail)
    fit = fit_gaussian_lines(prepared())["line1"]
    assert fit.status == "failed"
    assert "deliberate synthetic" in fit.message
    assert fit.area_angstrom is None


def test_nonfinite_optimizer_result_is_not_a_measurement(monkeypatch):
    def nonfinite(*args, **kwargs):
        return SimpleNamespace(success=True, x=np.array([np.nan, 0, 0.3]), nfev=1, message="bad")
    monkeypatch.setattr(module, "least_squares", nonfinite)
    fit = fit_gaussian_lines(prepared())["line1"]
    assert fit.status == "failed"
    assert fit.area_angstrom is None


def test_nonfinite_optimizer_residuals_are_not_a_measurement(monkeypatch):
    def nonfinite(*args, **kwargs):
        return SimpleNamespace(
            success=True, x=np.array([0.3, 0, 0.3]), nfev=1, message="bad",
            fun=np.array([np.inf]), jac=np.ones((1, 3)),
        )
    monkeypatch.setattr(module, "least_squares", nonfinite)
    fit = fit_gaussian_lines(prepared())["line1"]
    assert fit.status == "failed"
    assert fit.area_angstrom is None


def test_ill_conditioned_covariance_is_left_undefined_with_a_flag():
    fit = fit_gaussian_lines(
        prepared(noise=0.003), options=FitOptions(max_jacobian_condition=1.01),
    )["line1"]
    assert fit.status == "converged"
    assert "covariance_ill_conditioned" in fit.flags
    assert fit.covariance is None
    assert fit.area_error_angstrom is None


def test_low_amplitude_relative_to_declared_noise_requires_inspection():
    fit = fit_gaussian_lines(prepared(amplitude=0.005, uncertainty=0.01))["line1"]
    assert fit.status == "converged"
    assert "low_amplitude_snr" in fit.flags


def test_too_few_pixels_and_no_covariance_degrees_of_freedom():
    x = REST+np.array([-0.4, 0.07, 0.4])
    spectrum = prepare_spectrum(
        x, analytic(x), wavelength_unit="angstrom", wavelength_frame="air",
        continuum_normalised=True,
    )
    unavailable = fit_gaussian_lines(spectrum)["line1"]
    assert unavailable.status == "unavailable"
    assert unavailable.area_angstrom is None
    fitted = fit_gaussian_lines(spectrum, options=FitOptions(min_pixels=3))["line1"]
    assert fitted.status == "converged"
    assert "covariance_insufficient_dof" in fitted.flags
    assert fitted.area_error_angstrom is None


@pytest.mark.parametrize("keyword,value", [
    ("windows", {"line10": (1, 2)}), ("initial_parameters", {"bad": PARAMETERS}),
    ("excluded_regions", {"line0": ((1, 2),)}), ("excluded_lines", ["line10"]),
    ("excluded_lines", ["line1", "line1"]), ("windows", {"line1": (REST+2, REST-2)}),
    ("windows", {"line1": (float("nan"), REST)}),
    ("windows", {"line1": (REST+2, REST+3)}),
    ("excluded_regions", {"line1": ((REST+1, REST),)}),
    ("initial_parameters", {"line2": GaussianParameters(2, 21812.169, 0.3)}),
])
def test_invalid_overrides_are_rejected_even_for_lines_without_coverage(keyword, value):
    with pytest.raises(ValueError):
        fit_gaussian_lines(prepared(), **{keyword: value})


@pytest.mark.parametrize("keyword,value", [
    ("windows", []), ("excluded_regions", []), ("initial_parameters", []),
    ("initial_parameters", {"line1": (0.3, REST, 0.3)}),
    ("excluded_lines", "line1"), ("options", {}),
])
def test_invalid_override_types(keyword, value):
    with pytest.raises(TypeError):
        fit_gaussian_lines(prepared(), **{keyword: value})


@pytest.mark.parametrize("keyword,value", [
    ("min_pixels", 2), ("max_nfev", 0), ("window_half_width_angstrom", 0),
    ("sigma_bounds_angstrom", (0, 1)), ("amplitude_bounds", (-0.1, 1)),
    ("amplitude_bounds", (1, 1)), ("boundary_fraction", 0.5),
    ("gap_factor", 1), ("rms_warning", float("nan")),
])
def test_invalid_fitting_options(keyword, value):
    with pytest.raises(ValueError):
        FitOptions(**{keyword: value})


@pytest.mark.parametrize("keyword,value", [("min_pixels", True), ("max_nfev", 2.5)])
def test_fitting_options_require_integer_iteration_and_pixel_counts(keyword, value):
    with pytest.raises(TypeError):
        FitOptions(**{keyword: value})


@pytest.mark.parametrize("amplitude,sigma", [(-1, 1), (0.3, 0), (0.3, -1), (np.nan, 1)])
def test_invalid_gaussian_area_inputs(amplitude, sigma):
    with pytest.raises(ValueError):
        gaussian_area(amplitude, sigma)
