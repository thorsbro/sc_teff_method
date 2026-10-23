"""Independent numerical checks using newly constructed analytical spectra.

This module deliberately does not use the example spectrum generator or any
observational inputs. Mathematical profiles, derivatives and covariance are
calculated here independently of the fitting model implementation.
"""

import importlib
import math

import pytest

from sc_teff_method import LINE_IDS, estimate_temperature, get_calibration

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
fitting = importlib.import_module("sc_teff_method.fitting")
workflow = importlib.import_module("sc_teff_method.workflow")
spectrum_input = importlib.import_module("sc_teff_method.spectrum")

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}


def independent_line():
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    wavelength = np.linspace(rest - 3.4, rest + 3.4, 321)
    amplitude, centre, sigma = 0.28, rest + 0.17, 0.37
    flux = 1 - amplitude * np.exp(-(wavelength - centre)**2 / (2 * sigma**2))
    return wavelength, flux, amplitude, centre, sigma


def test_independent_nine_line_recovery_and_temperature_arithmetic():
    wavelengths, truth = [], {}
    for i, (line_id, line) in enumerate(get_calibration().lines.items()):
        wavelengths.append(np.linspace(
            line.wavelength_air_angstrom - 3.4, line.wavelength_air_angstrom + 3.4, 321,
        ))
        truth[line_id] = (
            0.21 + 0.021 * i,
            line.wavelength_air_angstrom + 0.11 * math.sin(i + 1),
            0.29 + 0.013 * i,
        )
    wavelength = np.concatenate(wavelengths)
    flux = np.ones(wavelength.shape)
    for amplitude, centre, sigma in truth.values():
        flux -= amplitude * np.exp(-(wavelength - centre)**2 / (2 * sigma**2))
    result = workflow.fit_spectrum(wavelength, flux, feh=-0.2, **CONVENTIONS)
    assert result.included_line_ids == LINE_IDS
    expected_areas = {}
    for line_id, (amplitude, centre, sigma) in truth.items():
        fit = result.fits[line_id]
        assert fit.status == "converged"
        assert not fit.flags
        assert fit.requires_inspection
        assert fit.parameters.amplitude == pytest.approx(amplitude, rel=0, abs=5e-9)
        assert fit.parameters.centre_air_angstrom == pytest.approx(centre, rel=0, abs=5e-9)
        assert fit.parameters.sigma_angstrom == pytest.approx(sigma, rel=0, abs=5e-9)
        expected_areas[line_id] = amplitude * sigma * (2 * math.pi)**0.5
        assert fit.area_angstrom == pytest.approx(expected_areas[line_id], rel=0, abs=5e-9)
    published = estimate_temperature(expected_areas, feh=-0.2)
    assert result.temperature.temperature_k == pytest.approx(
        published.temperature_k, rel=0, abs=1e-5,
    )
    calibration = get_calibration()
    independent_temperatures = [
        calibration.lines[line].a_k_per_angstrom * area
        + calibration.lines[line].b_k_per_dex * -0.2 + calibration.lines[line].c_k
        for line, area in expected_areas.items()
    ]
    assert result.temperature.temperature_k == pytest.approx(
        sum(independent_temperatures) / 9, rel=0, abs=1e-5,
    )
    assert result.performance.line_ids == LINE_IDS
    assert result.performance.cohort == "exact_subset"
    # Several invented strengths deliberately test preserved extrapolation notices.
    assert any(w.code == "outside_training_strength_range" for w in result.calibration_warnings)


@pytest.mark.parametrize("weighted", [False, True])
def test_independent_jacobian_and_covariance_with_area_cross_term(monkeypatch, weighted):
    wavelength, flux, amplitude, centre, sigma = independent_line()
    uncertainty = 0.002 + np.linspace(0, 0.003, wavelength.size)
    flux = flux + np.random.default_rng(196).normal(size=flux.size) * uncertainty
    spectrum = spectrum_input.prepare_spectrum(
        wavelength, flux, uncertainty=uncertainty if weighted else None, **CONVENTIONS,
    )
    original_optimizer = fitting.least_squares
    comparisons = []

    def compare_derivatives(function, initial, *, jac, **kwargs):
        parameters = np.array([amplitude, 0.17, sigma])
        perturbations = np.eye(3) * 1e-6
        finite_difference = np.column_stack([
            (function(parameters + delta) - function(parameters - delta)) / (2e-6)
            for delta in perturbations
        ])
        analytic = jac(parameters)
        comparisons.append(np.linalg.norm(analytic - finite_difference) / np.linalg.norm(analytic))
        return original_optimizer(function, initial, jac=jac, **kwargs)

    monkeypatch.setattr(fitting, "least_squares", compare_derivatives)
    fit = fitting.fit_gaussian_lines(spectrum)["line1"]
    assert comparisons and max(comparisons) < 3e-8
    assert fit.status == "converged" and not fit.flags
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    selected = (wavelength >= rest - 3) & (wavelength <= rest + 3)
    offsets = wavelength[selected] - rest
    fitted = fit.parameters
    parameters = np.array([
        fitted.amplitude, fitted.centre_air_angstrom - rest, fitted.sigma_angstrom,
    ])

    def independent_profile(values):
        return 1 - values[0] * np.exp(-(offsets - values[1])**2 / (2 * values[2]**2))

    jacobian = np.column_stack([
        (independent_profile(parameters + delta) - independent_profile(parameters - delta)) / (2e-6)
        for delta in np.eye(3) * 1e-6
    ])
    residuals = independent_profile(parameters) - flux[selected]
    if weighted:
        jacobian = jacobian / uncertainty[selected, None]
    expected_covariance = np.linalg.inv(jacobian.T @ jacobian)
    if not weighted:
        expected_covariance *= np.sum(residuals**2) / (len(residuals) - 3)
    difference = np.linalg.norm(np.asarray(fit.covariance) - expected_covariance)
    assert difference / np.linalg.norm(expected_covariance) < 2e-6
    gradient = (2 * math.pi)**0.5 * np.array([fitted.sigma_angstrom, 0, fitted.amplitude])
    expected_area_error = (gradient @ expected_covariance @ gradient)**0.5
    assert fit.area_error_angstrom == pytest.approx(expected_area_error, rel=2e-6)
    diagonal_only = np.sum(gradient**2 * np.diag(expected_covariance))**0.5
    assert expected_covariance[0, 2] < 0
    assert diagonal_only > expected_area_error * 1.2
    assert abs(fitted.amplitude - amplitude) < 5 * fit.parameter_errors.amplitude
    assert abs(fitted.centre_air_angstrom - centre) < 5 * fit.parameter_errors.centre_air_angstrom
    assert abs(fitted.sigma_angstrom - sigma) < 5 * fit.parameter_errors.sigma_angstrom


@pytest.mark.parametrize("rv_kms", [-90, 90])
def test_independent_redshift_and_blueshift_corrections(rv_kms):
    wavelength, flux, amplitude, centre, sigma = independent_line()
    beta = rv_kms / 299792.458
    factor = ((1 + beta) / (1 - beta))**0.5
    raw_wavelength = wavelength * factor
    corrected = workflow.fit_spectrum(
        raw_wavelength, flux, rv_kms=rv_kms, feh=-0.2, **CONVENTIONS,
    )
    fit = corrected.fits["line1"]
    assert corrected.included_line_ids == ("line1",)
    assert fit.parameters.centre_air_angstrom == pytest.approx(centre, rel=0, abs=5e-9)
    assert fit.area_angstrom == pytest.approx(amplitude * sigma * (2 * math.pi)**0.5, abs=5e-9)
    uncorrected = workflow.fit_spectrum(raw_wavelength, flux, feh=-0.2, **CONVENTIONS)
    assert uncorrected.temperature is None
    assert uncorrected.fits["line1"].status == "unavailable"
    assert uncorrected.fits["line1"].area_angstrom is None


@pytest.mark.parametrize("mode", ["gap", "mask"])
def test_gapped_or_masked_core_requires_explicit_warned_fit_inclusion(mode):
    wavelength, flux, amplitude, centre, sigma = independent_line()
    mask = np.abs(wavelength - centre) < 0.32
    if mode == "gap":
        wavelength, flux = wavelength[~mask], flux[~mask]
        options = {}
    else:
        flux = np.where(mask, np.nan, flux)
        options = {"mask": mask}
    result = workflow.fit_spectrum(wavelength, flux, feh=-0.2, **options, **CONVENTIONS)
    fit = result.fits["line1"]
    assert fit.status == "converged"
    assert "line_core_gap" in fit.flags
    assert ("internal_gap" if mode == "gap" else "masked_gap") in fit.flags
    assert not result.included_line_ids
    assert result.temperature is None and result.performance is None
    allowed = workflow.fit_spectrum(
        wavelength, flux, feh=-0.2, quality_policy="include_warned", **options, **CONVENTIONS,
    )
    assert allowed.included_line_ids == ("line1",)
    assert allowed.fits["line1"].area_angstrom == pytest.approx(
        amplitude * sigma * (2 * math.pi)**0.5, rel=0, abs=5e-9,
    )
    assert "line1" in allowed.line_temperatures_k
    assert "line2" not in allowed.line_temperatures_k


def test_optimizer_failure_cannot_be_included_even_with_warned_fit_policy():
    wavelength, flux, _, _, _ = independent_line()
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    result = workflow.fit_spectrum(
        wavelength, flux, feh=-0.2, quality_policy="include_warned",
        initial_parameters={"line1": fitting.GaussianParameters(0.7, rest - 0.4, 0.9)},
        options=fitting.FitOptions(max_nfev=1), **CONVENTIONS,
    )
    assert result.fits["line1"].status == "failed"
    assert result.fits["line1"].area_angstrom is None
    assert "line1" not in result.line_temperatures_k
    assert result.temperature is None and result.performance is None
