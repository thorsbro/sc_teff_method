"""HFS measurements use invented analytical mixtures, never observed spectra."""
# ruff: noqa: E402

import math
from dataclasses import FrozenInstanceError

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sc_teff_method import LINE_IDS
from sc_teff_method.continuum import ContinuumCorrection
from sc_teff_method.fitting import FitOptions, fit_pixel_masks
from sc_teff_method.hfs import hfs_absorption
from sc_teff_method.hfs_data import load_hfs_atomic_data
from sc_teff_method.hfs_fitting import HFSFitOptions, HFSParameters, fit_hfs_lines
from sc_teff_method.hfs_synthetic import HFS_SYNTHETIC_CASES, make_hfs_synthetic_spectrum
from sc_teff_method.spectrum import prepare_spectrum

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}


def prepared(case="clean"):
    example = make_hfs_synthetic_spectrum(case)
    return prepare_spectrum(
        example.wavelength_air_angstrom, example.flux, uncertainty=example.uncertainty,
        mask=example.mask, rv_kms=example.rv_kms, **CONVENTIONS,
    ), example


def isolated(*, line="line1", amplitude=0.45, sigma=0.31, offset=0.08,
             step=0.025, noise=0.0, supplied_error=None, continuum=None):
    atomic = load_hfs_atomic_data()
    definition = atomic.lines[line]
    rest = definition.wavelength_air_angstrom
    x = rest+np.linspace(-5, 5, round(10/step)+1)
    # Independent loop and exact rational wavelength offsets, without invoking
    # the package profile evaluator or solver to construct its test input.
    flux = np.ones(x.shape)
    total = math.fsum(c.strength for c in atomic.components[line])
    for component in atomic.components[line]:
        delta = -rest*component.offset_cm1/(definition.cog_wavenumber_cm1+component.offset_cm1)
        flux -= amplitude*(component.strength/total)*np.exp(
            -(x-(rest+offset)-delta)**2/(2*sigma**2)
        )
    if noise:
        flux += np.random.default_rng(631702).normal(0, noise, x.size)
    error = None if supplied_error is None else np.full(x.shape, supplied_error)
    if continuum is not None:
        flux *= continuum(x)
        if error is not None:
            error *= continuum(x)
    return prepare_spectrum(x, flux, uncertainty=error, **CONVENTIONS)


@pytest.mark.parametrize("case", HFS_SYNTHETIC_CASES)
def test_all_hfs_examples_are_deterministic_invented_read_only_arrays(case):
    first, second = make_hfs_synthetic_spectrum(case), make_hfs_synthetic_spectrum(case)
    assert first.seed == 20261025
    assert first.atomic_dataset_id == "hala2022_ab_generated"
    assert tuple(first.truth) == LINE_IDS
    np.testing.assert_array_equal(first.flux, second.flux)
    np.testing.assert_array_equal(first.wavelength_air_angstrom, second.wavelength_air_angstrom)
    assert not first.flux.flags.writeable
    assert not first.wavelength_air_angstrom.flags.writeable
    assert "No observational" in first.description


@pytest.mark.parametrize("line", LINE_IDS)
def test_nine_line_noiseless_recovery_uses_hfs_area_definition(line):
    spectrum, example = prepared()
    fits = fit_hfs_lines(spectrum)
    fit, truth = fits[line], example.truth[line]
    assert tuple(fits) == LINE_IDS
    assert fit.status == "converged"
    assert fit.measurement_method == "hfs"
    assert fit.n_components == len(load_hfs_atomic_data().components[line])
    assert fit.parameters.amplitude == pytest.approx(truth.amplitude, abs=1e-7)
    assert fit.parameters.centre_air_angstrom == pytest.approx(truth.centre_air_angstrom, abs=1e-7)
    assert fit.parameters.sigma_angstrom == pytest.approx(truth.sigma_angstrom, abs=1e-7)
    assert fit.area_angstrom == pytest.approx(
        math.sqrt(2*math.pi)*truth.amplitude*truth.sigma_angstrom, abs=1e-8,
    )
    assert fit.flags == ()
    assert fit.requires_inspection
    assert fit.weighting == "unweighted"
    with pytest.raises(FrozenInstanceError):
        fit.area_angstrom = 0
    with pytest.raises(TypeError):
        fits["line1"] = None


@pytest.mark.parametrize("amplitude,sigma,offset", [
    (0.15, 0.13, -0.2), (0.42, 0.4, 0.08), (0.8, 0.9, 0.25),
])
def test_artificial_parameter_recovery_for_narrow_and_broad_hfs_profiles(amplitude, sigma, offset):
    spectrum = isolated(amplitude=amplitude, sigma=sigma, offset=offset)
    fit = fit_hfs_lines(spectrum)["line1"]
    rest = fit.rest_wavelength_air_angstrom
    assert fit.status == "converged"
    assert fit.parameters.amplitude == pytest.approx(amplitude, abs=1e-7)
    assert fit.parameters.centre_air_angstrom == pytest.approx(rest+offset, abs=1e-7)
    assert fit.parameters.sigma_angstrom == pytest.approx(sigma, abs=1e-7)


def test_fixed_component_mixture_numerically_integrates_to_full_analytic_area():
    atomic = load_hfs_atomic_data()
    line = atomic.lines["line9"]
    x = line.wavelength_air_angstrom+np.linspace(-8, 8, 64001)
    flux = hfs_absorption(
        x, 0.64, line.wavelength_air_angstrom+0.04, 0.38, atomic.components["line9"],
        parent_wavelength=line.wavelength_air_angstrom, cog_wavenumber_cm1=line.cog_wavenumber_cm1,
    )
    expected = math.sqrt(2*math.pi)*0.64*0.38
    assert float(np.sum(1-flux)*0.00025) == pytest.approx(expected, abs=2e-12)


def test_seeded_noise_recovery_and_supplied_uncertainties_do_not_change_baseline_weights():
    first = fit_hfs_lines(isolated(noise=0.0025, supplied_error=0.0025))["line1"]
    second = fit_hfs_lines(isolated(noise=0.0025, supplied_error=0.02))["line1"]
    absent = fit_hfs_lines(isolated(noise=0.0025))["line1"]
    assert first.status == "converged"
    assert first.parameters.amplitude == pytest.approx(0.45, abs=0.005)
    assert first.parameters.sigma_angstrom == pytest.approx(0.31, abs=0.006)
    assert first.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*0.45*0.31, abs=0.005)
    assert first.parameters == second.parameters == absent.parameters
    assert first.area_error_angstrom == second.area_error_angstrom == absent.area_error_angstrom
    assert first.covariance == second.covariance == absent.covariance
    assert first.weighting == second.weighting == absent.weighting == "unweighted"
    assert first.reduced_chi2 is not None and second.reduced_chi2 is not None
    assert absent.reduced_chi2 is None
    assert first.reduced_chi2 == pytest.approx(64*second.reduced_chi2)


def test_formal_covariance_and_area_error_include_amplitude_width_covariance():
    spectrum = isolated(noise=0.0025)
    fit = fit_hfs_lines(spectrum)["line1"]
    _, used = fit_pixel_masks(spectrum, fit.window)
    x, y = spectrum.wavelength_air_angstrom[used], spectrum.flux[used]
    params = fit.parameters
    atomic = load_hfs_atomic_data()
    pattern = atomic.components["line1"]
    weights = np.array([component.strength for component in pattern])
    weights /= np.sum(weights)
    offsets = np.array(fit.component_offsets_angstrom)
    d = x[:, None]-params.centre_air_angstrom-offsets
    exponent = np.exp(-d**2/(2*params.sigma_angstrom**2))
    jacobian = np.column_stack((
        -(exponent@weights),
        -params.amplitude*((exponent*d)@weights)/params.sigma_angstrom**2,
        -params.amplitude*((exponent*d**2)@weights)/params.sigma_angstrom**3,
    ))
    model = 1-params.amplitude*(exponent@weights)
    residuals = model-y
    covariance = np.linalg.inv(jacobian.T@jacobian)*np.sum(residuals**2)/(len(x)-3)
    gradient = math.sqrt(2*math.pi)*np.array([params.sigma_angstrom, 0, params.amplitude])
    variance = float(gradient@covariance@gradient)
    np.testing.assert_allclose(fit.covariance, covariance, rtol=2e-8, atol=1e-15)
    assert fit.area_error_angstrom == pytest.approx(math.sqrt(variance), rel=2e-8)
    diagonal_variance = float(gradient@np.diag(np.diag(covariance))@gradient)
    assert abs(variance-diagonal_variance) > variance*0.05


def test_missing_fully_masked_and_explicitly_excluded_lines_remain_missing():
    spectrum, _ = prepared("missing")
    fits = fit_hfs_lines(spectrum, excluded_lines=["line2"])
    assert fits["line2"].status == "excluded"
    for line in ("line3", "line7"):
        assert fits[line].status == "unavailable"
        assert fits[line].area_angstrom is None
        assert fits[line].parameters is None
        assert "no_coverage" in fits[line].flags
    complete = isolated()
    masked = prepare_spectrum(complete.wavelength_air_angstrom, complete.flux,
                              mask=np.ones(complete.n_pixels, dtype=bool), **CONVENTIONS)
    fit = fit_hfs_lines(masked)["line1"]
    assert fit.status == "unavailable"
    assert fit.n_pixels == 0
    assert fit.area_angstrom is None
    assert "insufficient_samples" in fit.flags


def test_mask_intervals_and_input_masks_are_preserved_in_exact_native_coordinates():
    spectrum = isolated()
    rest = load_hfs_atomic_data().lines["line1"].wavelength_air_angstrom
    mask = (spectrum.wavelength_air_angstrom >= rest+1.4) & (
        spectrum.wavelength_air_angstrom <= rest+1.8
    )
    masked = prepare_spectrum(spectrum.wavelength_air_angstrom, spectrum.flux,
                              mask=mask, **CONVENTIONS)
    window, regions = (rest-2.8, rest+2.8), ((rest-1.8, rest-1.3),)
    fit = fit_hfs_lines(masked, windows={"line1": window},
                        excluded_regions={"line1": regions})["line1"]
    in_window, used = fit_pixel_masks(masked, window, regions)
    assert fit.n_pixels == int(used.sum())
    assert fit.n_excluded == int((in_window & ~used).sum())
    assert fit.window == window
    assert fit.excluded_regions == regions
    assert "masked_gap" in fit.flags
    assert fit.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*0.45*0.31, abs=1e-8)


@pytest.mark.parametrize("velocity", [-67.0, 67.0])
def test_explicit_rv_correction_changes_only_wavelength_coordinates_once(velocity):
    original = isolated()
    beta = velocity/299792.458
    factor = math.sqrt((1+beta)/(1-beta))
    raw = original.wavelength_air_angstrom*factor
    restored = prepare_spectrum(raw, original.flux, rv_kms=velocity, **CONVENTIONS)
    fit = fit_hfs_lines(restored)["line1"]
    reference = fit_hfs_lines(original)["line1"]
    assert fit.area_angstrom == pytest.approx(reference.area_angstrom, abs=2e-10)
    assert fit.parameters.centre_air_angstrom == pytest.approx(
        reference.parameters.centre_air_angstrom, abs=1e-8,
    )
    assert restored.rv_kms == velocity
    assert restored.doppler_factor == factor


@pytest.mark.parametrize("mode", ["constant", "linear"])
def test_local_continuum_is_applied_from_original_once_and_preserves_other_lines(mode):
    base, _ = prepared("noise")
    atomic = load_hfs_atomic_data()
    rest = atomic.lines["line2"].wavelength_air_angstrom
    x = base.wavelength_air_angstrom
    region = np.abs(x-rest) <= 5
    level = np.ones(x.shape)
    level[region] = 1.035+(0.006*(x[region]-rest) if mode == "linear" else 0)
    spectrum = prepare_spectrum(x, base.flux*level, uncertainty=0.002*level, **CONVENTIONS)
    correction = (ContinuumCorrection(enabled=True, mode="constant", level=1.035)
                  if mode == "constant" else ContinuumCorrection(
                      enabled=True, mode="linear", anchors=((rest-3, 1.017), (rest+3, 1.053)),
                  ))
    first = fit_hfs_lines(spectrum)
    corrected = fit_hfs_lines(spectrum, continuum_corrections={"line2": correction})
    repeated = fit_hfs_lines(spectrum, continuum_corrections={"line2": correction})
    baseline = prepare_spectrum(x, base.flux, uncertainty=np.full(x.shape, 0.002), **CONVENTIONS)
    expected = fit_hfs_lines(baseline)["line2"]
    assert corrected["line2"].area_angstrom == pytest.approx(expected.area_angstrom, abs=1e-9)
    assert corrected["line2"].parameters == repeated["line2"].parameters
    assert corrected["line2"].reduced_chi2 == pytest.approx(expected.reduced_chi2, rel=1e-8)
    for line in LINE_IDS:
        if line != "line2":
            assert corrected[line] == first[line]
    np.testing.assert_array_equal(spectrum.flux, base.flux*level)
    assert not spectrum.flux.flags.writeable


def test_component_centres_near_or_outside_windows_have_distinct_flags():
    spectrum = isolated(line="line1", amplitude=0.55, sigma=0.13, step=0.005)
    rest = load_hfs_atomic_data().lines["line1"].wavelength_air_angstrom
    baseline = fit_hfs_lines(spectrum)["line1"]
    centres = baseline.parameters.centre_air_angstrom+np.array(baseline.component_offsets_angstrom)
    lower = float(centres.min()+0.003)
    fit = fit_hfs_lines(
        spectrum, windows={"line1": (lower, rest+1.5)},
        initial_parameters={"line1": HFSParameters(0.55, rest+0.08, 0.13)},
    )["line1"]
    assert fit.status == "converged"
    assert "hfs_components_cross_window" in fit.flags
    edge = float(centres.min())
    near = fit_hfs_lines(
        spectrum, windows={"line1": (edge, rest+1.5)},
        initial_parameters={"line1": HFSParameters(0.55, rest+0.08, 0.13)},
    )["line1"]
    assert "hfs_components_near_window_edge" in near.flags


def test_residual_bounds_sampling_and_failures_are_explicit():
    broad = fit_hfs_lines(isolated(sigma=0.8), options=HFSFitOptions(
        sigma_bounds_angstrom=(0.03, 0.4),
    ))["line1"]
    assert "sigma_near_bound" in broad.flags
    assert "formal_covariance_at_boundary" in broad.flags
    assert "high_residual_rms" in broad.flags
    coarse = fit_hfs_lines(isolated(sigma=0.08, step=0.4))["line1"]
    assert "poor_fwhm_sampling" in coarse.flags
    failed = fit_hfs_lines(isolated(), options=HFSFitOptions(max_nfev=1))["line1"]
    assert failed.status == "failed"
    assert failed.area_angstrom is None
    assert failed.parameters is None


def test_condition_threshold_is_configurable_without_changing_the_fit_or_covariance():
    spectrum = isolated(noise=0.0025)
    baseline = fit_hfs_lines(spectrum)["line1"]
    cautious = fit_hfs_lines(spectrum, options=HFSFitOptions(
        max_jacobian_condition=1.00001,
    ))["line1"]
    assert "ill_conditioned_covariance" not in baseline.flags
    assert "ill_conditioned_covariance" in cautious.flags
    assert cautious.parameters == baseline.parameters
    assert cautious.covariance == baseline.covariance
    assert cautious.area_error_angstrom == baseline.area_error_angstrom


@pytest.mark.parametrize("kwargs", [
    {"weighting": "inverse_sigma"}, {"weighting": True},
    {"windows": {"unknown": (1, 2)}}, {"windows": {"line1": (2, 1)}},
    {"initial_parameters": {"line1": (0.3, 22000, 0.3)}},
    {"excluded_lines": "line1"}, {"excluded_lines": ["line1", "line1"]},
    {"excluded_regions": {"line1": [(1, float("nan"))]}},
    {"continuum_corrections": {"line1": 1.0}},
    {"continuum_options": {"max_deviation_warning": 0.1}},
    {"options": "default"}, {"options": FitOptions(min_pixels=3)},
    {"options": FitOptions(amplitude_bounds=(0, 1.1))},
])
def test_invalid_input_overrides_fail_before_fitting(kwargs):
    with pytest.raises((TypeError, ValueError)):
        fit_hfs_lines(isolated(), **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"amplitude_bounds": (0, 1.1)}, {"min_pixels": 3},
])
def test_hfs_options_restrict_physical_and_formal_error_bounds(kwargs):
    with pytest.raises(ValueError):
        HFSFitOptions(**kwargs)


def test_invalid_continuum_is_not_silently_ignored_for_unavailable_or_excluded_line():
    spectrum = isolated()
    rest = load_hfs_atomic_data().lines["line9"].wavelength_air_angstrom
    invalid = ContinuumCorrection(
        enabled=True, mode="linear", anchors=((rest+2, 1), (rest+3, 2)),
    )
    with pytest.raises(ValueError, match="positive"):
        fit_hfs_lines(spectrum, continuum_corrections={"line9": invalid},
                      excluded_lines=["line9"])
