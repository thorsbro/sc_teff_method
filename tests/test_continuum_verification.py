"""Independent continuum checks: analytical Gaussian functions, never observations."""
# ruff: noqa: E402

import hashlib
import io
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sc_teff_method import LINE_IDS, estimate_temperature, get_calibration
from sc_teff_method.continuum import ContinuumCorrection
from sc_teff_method.fitting import fit_gaussian_lines
from sc_teff_method.review import ReviewSession
from sc_teff_method.review_io import load_spectrum_identity
from sc_teff_method.spectrum import prepare_spectrum
from sc_teff_method.synthetic import make_continuum_demo

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}
CSV_SETTINGS = {
    "wavelength_column": "wavelength", "flux_column": "flux", "uncertainty_column": None,
    "mask_column": None, "delimiter": ",", "header": True, "skiprows": 0,
}


def memory_identity(spectrum):
    return {
        "filename": "analytical.csv",
        "sha256": hashlib.sha256(spectrum.flux.tobytes()).hexdigest(),
        "loader_settings": CSV_SETTINGS,
    }


def demo_session(*, configuration="nandakumar_feh", noise_sigma=0.0):
    toy = make_continuum_demo(noise_sigma=noise_sigma)
    spectrum = prepare_spectrum(
        toy.wavelength_air_angstrom, toy.flux, uncertainty=toy.uncertainty, mask=toy.mask,
        **CONVENTIONS,
    )
    session = ReviewSession.create(
        spectrum, input_identity=memory_identity(spectrum), configuration=configuration,
        feh=-0.2 if configuration == "nandakumar_feh" else None,
    )
    return session, toy


def prescribe(session, toy, line):
    truth = toy.continuum_truth[line]
    return session.edit_line(
        line, continuum=ContinuumCorrection(enabled=truth.mode != "none", **asdict(truth)),
    )


def test_new_continuum_demo_is_analytical_deterministic_and_does_not_hide_coverage():
    first, second = make_continuum_demo(), make_continuum_demo()
    assert first.seed == 20261024
    assert first.uncertainty is None
    assert set(first.truth) == set(first.continuum_truth) == set(LINE_IDS)
    assert first.continuum_truth["line1"].mode == "none"
    assert first.continuum_truth["line2"].level == 1.035
    assert first.continuum_truth["line3"].mode == "linear"
    assert first.mask.any()
    assert not first.flux.flags.writeable
    np.testing.assert_array_equal(first.flux, second.flux)
    noisy = make_continuum_demo(noise_sigma=0.002)
    same_noise = make_continuum_demo(noise_sigma=0.002)
    np.testing.assert_array_equal(noisy.flux, same_noise.flux)
    assert np.max(np.abs(noisy.flux-first.flux)) > 0.001
    assert np.all(noisy.uncertainty > 0)
    session, _ = demo_session()
    assert "high_residual_rms" in session.fits["line4"].flags
    assert "masked_gap" in session.fits["line7"].flags
    assert "partial_window_coverage" in session.fits["line8"].flags


@pytest.mark.parametrize("noise_sigma", [-1, float("nan"), float("inf"), True, "0.1", 1j])
def test_synthetic_noise_is_explicit_and_validated(noise_sigma):
    with pytest.raises((TypeError, ValueError)):
        make_continuum_demo(noise_sigma=noise_sigma)


@pytest.mark.parametrize("configuration", ["nandakumar_feh", "nandakumar_no_feh"])
def test_prescribed_continua_recover_independent_gaussian_areas_and_temperatures(configuration):
    session, toy = demo_session(configuration=configuration)
    selected = ("line1", "line2", "line3", "line6", "line9")
    assert session.reviewed.temperature is None
    for line in selected:
        fit = prescribe(session, toy, line)
        truth = toy.truth[line]
        analytical_area = math.sqrt(2*math.pi) * truth.amplitude * truth.sigma_angstrom
        assert fit.status == "converged"
        assert fit.parameters.amplitude == pytest.approx(truth.amplitude, abs=2e-10)
        assert fit.parameters.centre_air_angstrom == pytest.approx(
            truth.centre_air_angstrom, abs=2e-9,
        )
        assert fit.parameters.sigma_angstrom == pytest.approx(truth.sigma_angstrom, abs=2e-10)
        assert fit.area_angstrom == pytest.approx(analytical_area, abs=2e-10)
        session.decide(line, "accepted", note="Inspected the prescribed analytical continuum.")
    expected = estimate_temperature(
        {line: math.sqrt(2*math.pi)*toy.truth[line].amplitude*toy.truth[line].sigma_angstrom
         for line in selected},
        configuration=configuration, feh=session.feh,
    )
    assert session.reviewed.temperature.temperature_k == pytest.approx(
        expected.temperature_k, abs=2e-7,
    )
    assert session.reviewed.temperature.temperature_k == pytest.approx(
        sum(session.reviewed.line_temperatures_k[line] for line in selected)/len(selected),
        abs=1e-12,
    )


@pytest.mark.parametrize("mode", ["constant", "linear"])
@pytest.mark.parametrize("rv_kms", [-73.0, 73.0])
def test_independent_correction_preserves_weights_masks_and_existing_rv(mode, rv_kms):
    rest = get_calibration().lines["line2"].wavelength_air_angstrom
    x = rest + np.linspace(-4, 4, 501)
    amplitude, centre, width = 0.37, rest-0.085, 0.41
    reference_flux = 1-amplitude*np.exp(-(x-centre)**2/(2*width**2))
    prescribed = 1.045 + (0.009*(x-rest) if mode == "linear" else np.zeros(x.shape))
    error = (0.0018+0.0001*np.cos(x-rest))*prescribed
    mask = (x >= rest+1.8) & (x <= rest+2.0)
    factor = math.sqrt((1+rv_kms/299792.458)/(1-rv_kms/299792.458))
    original = prepare_spectrum(
        x*factor, reference_flux*prescribed, uncertainty=error, mask=mask,
        rv_kms=rv_kms, **CONVENTIONS,
    )
    session = ReviewSession.create(original, input_identity=memory_identity(original), feh=-0.2)
    correction = (
        ContinuumCorrection(enabled=True, mode="constant", level=1.045)
        if mode == "constant" else ContinuumCorrection(
            enabled=True, mode="linear", anchors=((rest-3, 1.018), (rest+3, 1.072)),
        )
    )
    extra_mask = ((rest-1.6, rest-1.3),)
    measured = session.edit_line("line2", continuum=correction, excluded_regions=extra_mask)
    working = session.corrected_spectrum("line2")
    region = (x >= rest-3) & (x <= rest+3)
    np.testing.assert_allclose(working.flux[region], reference_flux[region], atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        working.uncertainty[region], error[region]/prescribed[region], atol=1e-14, rtol=0,
    )
    np.testing.assert_array_equal(working.flux[~region], original.flux[~region])
    np.testing.assert_array_equal(working.wavelength_air_angstrom, original.wavelength_air_angstrom)
    np.testing.assert_array_equal(working.mask, original.mask)
    assert working.rv_kms == original.rv_kms == rv_kms
    assert working.doppler_factor == original.doppler_factor == factor
    expected = prepare_spectrum(
        x*factor, reference_flux, uncertainty=error/prescribed, mask=mask,
        rv_kms=rv_kms, **CONVENTIONS,
    )
    direct = fit_gaussian_lines(
        expected, excluded_regions={"line2": extra_mask},
        excluded_lines=[line for line in LINE_IDS if line != "line2"],
    )["line2"]
    assert measured.area_angstrom == pytest.approx(direct.area_angstrom, abs=2e-11)
    assert measured.area_error_angstrom == pytest.approx(direct.area_error_angstrom, abs=2e-11)
    assert measured.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*amplitude*width, abs=1e-10)
    assert "masked_gap" in measured.flags


def test_continuum_refits_start_from_original_and_leave_other_eight_cached():
    session, toy = demo_session()
    session.decide("line1", "accepted")
    session.decide("line2", "accepted")
    before = dict(session.fits)
    original_flux = np.array(session.spectrum.flux)
    corrected = prescribe(session, toy, "line2")
    assert session.decisions["line2"].state == "pending"
    assert session.decisions["line1"].state == "accepted"
    for line in LINE_IDS:
        if line != "line2":
            assert session.fits[line] is before[line]
    session.decide("line2", "accepted")
    for _ in range(3):
        repeated = session.refit_line("line2")
        assert repeated == corrected
        assert session.decisions["line2"].state == "pending"
    assert session.uncorrected_fits["line2"] == before["line2"]
    np.testing.assert_array_equal(session.spectrum.flux, original_flux)
    session.reset_continuum("line2")
    assert session.fits["line2"] == before["line2"]
    assert session.decisions["line2"].state == "pending"


def test_incorrect_anchors_change_area_and_same_subset_temperature_without_validation_claim():
    session, toy = demo_session()
    for line in ("line1", "line2", "line3", "line5"):
        prescribe(session, toy, line)
        session.decide(line, "accepted")
    area = session.fits["line5"].area_angstrom
    temperature = session.reviewed.temperature.temperature_k
    session.edit_line("line5", continuum=ContinuumCorrection(
        enabled=True, mode="linear", anchors=toy.incorrect_anchor_example,
    ))
    assert session.decisions["line5"].state == "pending"
    assert "line5" not in session.reviewed.included_line_ids
    session.decide("line5", "accepted", note="Deliberately wrong invented anchors for comparison.")
    assert abs(session.fits["line5"].area_angstrom-area) > 0.02
    assert abs(session.reviewed.temperature.temperature_k-temperature) > 1
    warnings = " ".join(session.reviewed.warnings).lower()
    assert "continuum" in warnings and "do not validate" in warnings
    assert session.reviewed.performance.cohort == "exact_subset"


def legacy_source(path):
    """Rebuild the fixed fixture's spectrum from a formula, not an array fixture."""
    parts = []
    for i, (_, reference) in enumerate(get_calibration().lines.items()):
        rest = reference.wavelength_air_angstrom
        x = np.round(rest+np.linspace(-4, 4, 321), 6)
        flux = 1-(0.3+0.013*i)*np.exp(-0.5*((x-(rest+0.04))/(0.34+0.017*(i % 3)))**2)
        parts.append(np.column_stack((x, flux)))
    stream = io.StringIO()
    # Decimal rounding fixes the raw-file checksum across vectorised exp kernels.
    np.savetxt(
        stream, np.concatenate(parts), delimiter=",", header="wavelength,flux", comments="",
        fmt=("%.6f", "%.12f"),
    )
    path.write_text(stream.getvalue(), encoding="utf-8")
    return path


def test_fixed_public_pr3_schema1_review_reproduces_its_approved_numbers(tmp_path):
    fixture = Path(__file__).parent / "fixtures/continuum_pr3_review_v1.json"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "2ac37c03ab1268fdfe9c067d06261a3a1520220c591adb533370d026ecb5c3c2"
    )
    source = legacy_source(tmp_path / "analytical.csv")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "b22ba85a41bcfbdec022ddc0d188776eabaf77e278c890e3c4a25356746f0c0b"
    )
    spectrum, identity = load_spectrum_identity(source, loader_settings=CSV_SETTINGS, **CONVENTIONS)
    original = json.loads(fixture.read_text())
    assert original["schema_version"] == 1
    session = ReviewSession.load(spectrum, fixture, input_identity=identity)
    assert session.reviewed.included_line_ids == ("line1", "line3", "line9")
    assert session.reviewed.temperature.temperature_k == pytest.approx(3634.6822067466187, abs=2e-8)
    assert session.decisions["line7"].state == "rejected"
    for line in LINE_IDS:
        assert session.controls[line].continuum == ContinuumCorrection()
        expected = original["lines"][line]["fit"]["area_angstrom"]
        assert session.fits[line].area_angstrom == pytest.approx(expected, abs=1e-10)
    upgraded = session.to_dict()
    assert upgraded["schema_version"] == 2
    migrated = ReviewSession.from_dict(spectrum, upgraded, input_identity=identity)
    assert migrated.reviewed.temperature == session.reviewed.temperature


def test_approved_continuum_controls_are_fingerprinted_even_if_result_stays_unity():
    session, _ = demo_session()
    session.decide("line1", "accepted")
    payload = session.to_dict()
    controls = payload["lines"]["line1"]["controls"]["continuum"]
    controls.update({"enabled": True, "mode": "constant", "level": 1.0})
    with pytest.raises(ValueError, match="[Cc]ontrol|approval"):
        ReviewSession.from_dict(
            session.spectrum, payload, input_identity=dict(session.input_identity),
        )


def test_saved_continuum_review_and_headless_replay_recompute_accepted_results(tmp_path):
    pytest.importorskip("matplotlib")
    toy = make_continuum_demo(noise_sigma=0.0012)
    source = tmp_path / "synthetic.csv"
    np.savetxt(
        source,
        np.column_stack((
            toy.wavelength_air_angstrom, toy.flux, toy.uncertainty, toy.mask.astype(int),
        )),
        delimiter=",", header="wavelength,flux,error,exclude", comments="", fmt="%.17g",
    )
    settings = {**CSV_SETTINGS, "uncertainty_column": "error", "mask_column": "exclude"}
    spectrum, identity = load_spectrum_identity(source, loader_settings=settings, **CONVENTIONS)
    session = ReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    for line in ("line1", "line2", "line3", "line7"):
        prescribe(session, toy, line)
        session.decide(line, "accepted", note="Inspected the known synthetic continuum.")
    session.decide("line8", "rejected", note="Deliberate incomplete coverage.")
    review = tmp_path / "review.json"
    session.save(review)
    saved = json.loads(review.read_text())
    assert saved["schema_version"] == 2
    assert saved["lines"]["line3"]["controls"]["continuum"]["mode"] == "linear"
    assert not {"flux", "wavelength_air_angstrom", "corrected_flux"} & set(saved["input"])
    loaded = ReviewSession.load(spectrum, review, input_identity=identity)
    assert loaded.reviewed.temperature == session.reviewed.temperature
    assert loaded.controls == session.controls
    final = tmp_path / "headless"
    process = subprocess.run(
        [sys.executable, "-m", "sc_teff_method.cli", "replay", str(source),
         "--review-file", str(review), "--output-dir", str(final),
         "--label", "Entirely analytical local continuum"],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads((final / "summary.json").read_text())
    assert result["temperature"]["temperature_k"] == session.reviewed.temperature.temperature_k
    assert result["included_line_ids"] == ["line1", "line2", "line3", "line7"]
    assert result["fits"]["line3"]["area_angstrom"] == session.fits["line3"].area_angstrom
    assert result["fits"]["line3"]["continuum"]["mode"] == "linear"
    assert result["fits"]["line2"]["continuum"]["level"] == 1.035
    assert result["fits"]["line3"]["uncorrected_area_angstrom"] == (
        session.uncorrected_fits["line3"].area_angstrom
    )
    assert (final / "diagnostics.png").read_bytes().startswith(b"\x89PNG")
