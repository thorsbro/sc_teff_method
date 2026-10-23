"""Analytical examples never use observed spectra or observed templates."""

import csv
import importlib.util
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sc_teff_method import LINE_IDS, get_calibration

np = pytest.importorskip("numpy")
synthetic = pytest.importorskip("sc_teff_method.synthetic")
SYNTHETIC_CASES = synthetic.SYNTHETIC_CASES
make_synthetic_spectrum = synthetic.make_synthetic_spectrum


@pytest.mark.parametrize("name", SYNTHETIC_CASES)
def test_synthetic_cases_are_deterministic_and_valid(name):
    first = make_synthetic_spectrum(name)
    second = make_synthetic_spectrum(name)
    assert first.name == name
    assert first.seed == 20261023
    assert tuple(first.truth) == LINE_IDS
    np.testing.assert_array_equal(first.wavelength_air_angstrom, second.wavelength_air_angstrom)
    np.testing.assert_array_equal(first.flux, second.flux)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert np.all(np.diff(first.wavelength_air_angstrom) > 0)
    assert np.all(np.isfinite(first.flux))
    assert first.mask.dtype == bool
    assert first.mask.shape == first.flux.shape == first.wavelength_air_angstrom.shape
    assert not first.mask.any()
    assert "Analytical" in first.description


def test_clean_model_and_area_are_independently_calculated():
    case = make_synthetic_spectrum("clean")
    expected_flux = np.ones(case.flux.shape)
    for i, line_id in enumerate(LINE_IDS):
        line = case.truth[line_id]
        rest = get_calibration().lines[line_id].wavelength_air_angstrom
        assert line.amplitude == 0.42 + 0.018 * i
        assert line.sigma_angstrom == 0.42 + 0.025 * (i % 3)
        assert line.centre_air_angstrom == rest + 0.06 * ((i % 3) - 1)
        expected_flux -= line.amplitude * np.exp(
            -(case.wavelength_air_angstrom - line.centre_air_angstrom)**2
            / (2 * line.sigma_angstrom**2)
        )
        # Independent Gaussian integral: A*sigma*sqrt(2*pi).
        assert line.area_angstrom == pytest.approx(
            line.amplitude * line.sigma_angstrom * (2 * math.pi)**0.5, rel=1e-15,
        )
        selected = np.abs(case.wavelength_air_angstrom - rest) <= 5
        x = case.wavelength_air_angstrom[selected]
        absorption = 1 - case.flux[selected]
        trapezoid = np.sum(np.diff(x) * (absorption[:-1] + absorption[1:]) / 2)
        assert trapezoid == pytest.approx(line.area_angstrom, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(case.flux, expected_flux, rtol=0, atol=2e-16)
    assert case.uncertainty is None


def test_known_independent_noise_changes_only_flux_and_has_declared_scale():
    clean = make_synthetic_spectrum("clean")
    noisy = make_synthetic_spectrum("noise")
    another_seed = make_synthetic_spectrum("noise", seed=23)
    np.testing.assert_array_equal(clean.wavelength_air_angstrom, noisy.wavelength_air_angstrom)
    np.testing.assert_array_equal(noisy.uncertainty, np.full(noisy.flux.shape, 0.003))
    assert clean.truth == noisy.truth
    assert not np.array_equal(noisy.flux, another_seed.flux)
    noise = noisy.flux - clean.flux
    assert np.mean(noise) == pytest.approx(0, abs=0.0002)
    assert np.std(noise) == pytest.approx(0.003, rel=0.06)
    expected_noise = np.random.default_rng(20261023).normal(0, 0.003, noisy.flux.size)
    np.testing.assert_allclose(noise, expected_noise, rtol=0, atol=1.2e-16)


def test_missing_case_omits_coverage_without_zero_filling_or_removing_truth():
    case = make_synthetic_spectrum("missing")
    for line_id, line in get_calibration().lines.items():
        pixels = np.abs(case.wavelength_air_angstrom - line.wavelength_air_angstrom) <= 5
        assert bool(pixels.any()) == (line_id not in ("line3", "line7"))
    assert tuple(case.truth) == LINE_IDS
    assert len(case.flux) == 7 * 251


def test_rv_offset_uses_positive_recession_and_preserves_rest_truth():
    clean = make_synthetic_spectrum("clean")
    shifted = make_synthetic_spectrum("rv_offset")
    beta = 65 / 299792.458
    factor = ((1 + beta) / (1 - beta))**0.5
    assert shifted.rv_kms == 65
    np.testing.assert_allclose(
        shifted.wavelength_air_angstrom / factor, clean.wavelength_air_angstrom, rtol=2e-16,
    )
    assert np.all(shifted.wavelength_air_angstrom > clean.wavelength_air_angstrom)
    np.testing.assert_array_equal(shifted.flux, clean.flux)
    assert shifted.truth == clean.truth


def test_blends_and_continuum_deviation_are_explicitly_artificial():
    clean = make_synthetic_spectrum("clean")
    blended = make_synthetic_spectrum("blends")
    np.testing.assert_array_equal(blended.wavelength_air_angstrom, clean.wavelength_air_angstrom)
    difference = blended.flux - clean.flux
    rest = get_calibration().lines
    near_line4_blend = np.abs(
        clean.wavelength_air_angstrom - (rest["line4"].wavelength_air_angstrom + .75)
    )
    assert difference[np.argmin(near_line4_blend)] < -0.22
    near_continuum = (
        np.abs(clean.wavelength_air_angstrom - rest["line6"].wavelength_air_angstrom) < 4
    )
    assert np.all(difference[near_continuum] > 0.02)
    assert blended.truth == clean.truth


def test_width_and_sampling_are_independently_varied_including_undersampling():
    variable = make_synthetic_spectrum("variable")
    steps = []
    widths = []
    for line_id, reference in get_calibration().lines.items():
        selected = np.abs(variable.wavelength_air_angstrom - reference.wavelength_air_angstrom) <= 5
        steps.append(float(np.median(np.diff(variable.wavelength_air_angstrom[selected]))))
        widths.append(variable.truth[line_id].sigma_angstrom)
    assert len({round(step, 3) for step in steps}) == 9
    assert len(set(widths)) > 5
    assert steps[-1] > 2 * variable.truth["line9"].sigma_angstrom


def test_generated_arrays_and_truth_are_immutable():
    case = make_synthetic_spectrum()
    with pytest.raises(ValueError, match="read-only"):
        case.flux[0] = 0
    with pytest.raises(TypeError):
        case.truth["line1"] = None
    with pytest.raises(FrozenInstanceError):
        case.truth["line1"].amplitude = 0


@pytest.mark.parametrize("case", ["observed", "", None, []])
def test_invalid_case_names(case):
    with pytest.raises(ValueError, match="case"):
        make_synthetic_spectrum(case)


@pytest.mark.parametrize("seed", [True, "123", .3, None])
def test_invalid_seed_types(seed):
    with pytest.raises(TypeError, match="seed"):
        make_synthetic_spectrum(seed=seed)


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="seed"):
        make_synthetic_spectrum(seed=-1)


def test_csv_example_is_precise_self_contained_and_refuses_overwrite(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/generate_synthetic_spectra.py"
    spec = importlib.util.spec_from_file_location("_generate_synthetic_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "generated"
    paths = module.write_examples(output)
    assert len(paths) == 6
    manifest = json.loads((output / "truth.json").read_text())
    assert set(manifest["cases"]) == set(SYNTHETIC_CASES)
    for name in SYNTHETIC_CASES:
        case = make_synthetic_spectrum(name)
        with (output / f"{name}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        np.testing.assert_array_equal(
            [float(row["wavelength"]) for row in rows], case.wavelength_air_angstrom,
        )
        np.testing.assert_array_equal([float(row["flux"]) for row in rows], case.flux)
        assert all(row["exclude"] == "0" for row in rows)
        assert manifest["cases"][name]["rv_kms"] == case.rv_kms
        assert manifest["cases"][name]["truth"]["line1"]["area_angstrom"] == (
            case.truth["line1"].area_angstrom
        )
    original = (output / "clean.csv").read_bytes()
    with pytest.raises(FileExistsError):
        module.write_examples(output)
    assert (output / "clean.csv").read_bytes() == original
