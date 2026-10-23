"""Manual continuum arithmetic and review persistence, entirely mathematical data."""
# ruff: noqa: E402

import copy
import hashlib
import json
import math
from dataclasses import asdict

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sc_teff_method import LINE_IDS, get_calibration
from sc_teff_method.continuum import (
    ContinuumCorrection,
    ContinuumOptions,
    apply_continuum,
    evaluate_continuum,
)
from sc_teff_method.fitting import fit_gaussian_lines
from sc_teff_method.review import ReviewSession, read_review
from sc_teff_method.spectrum import prepare_spectrum


def toy(kind="constant", *, rv=None, masked=False):
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    x = rest + np.linspace(-4, 4, 401)
    y = 1 - 0.31 * np.exp(-0.5 * ((x-rest-0.08)/0.34)**2)
    if kind == "linear":
        c = 1.03 + (x-rest)*0.012
        correction = ContinuumCorrection(
            enabled=True, mode="linear", anchors=((rest-3, 0.994), (rest+3, 1.066)),
        )
    elif kind == "constant":
        c = np.full_like(x, 1.04)
        correction = ContinuumCorrection(enabled=True, mode="constant", level=1.04)
    else:
        c = np.ones_like(x)
        correction = ContinuumCorrection()
    mask = np.zeros(x.size, dtype=bool)
    if masked:
        mask[(x > rest+1.5) & (x < rest+1.8)] = True
    factor = 1 if rv is None else math.sqrt((1+rv/299792.458)/(1-rv/299792.458))
    spectrum = prepare_spectrum(
        x*factor, y*c, uncertainty=0.003*c, mask=mask, rv_kms=rv,
        wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
    )
    identity = {"filename": "invented.csv", "sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "loader_settings": {}}
    session = ReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    return session, correction, c, y


def identity(session):
    result = dict(session.input_identity)
    result["loader_settings"] = dict(result["loader_settings"])
    return result


@pytest.mark.parametrize("correction", [
    ContinuumCorrection(), ContinuumCorrection(enabled=True),
    ContinuumCorrection(enabled=False, mode="constant", level=1.5),
])
def test_no_correction_is_identity_and_reuses_original_spectrum(correction):
    session, _, _, _ = toy("unity")
    original = session.spectrum
    working, flags = apply_continuum(original, correction, window=session.fits["line1"].window)
    assert working is original
    assert not flags
    fit = session.edit_line("line1", continuum=correction)
    direct = fit_gaussian_lines(original)["line1"]
    assert fit == direct
    assert session.uncorrected_fits["line1"] is fit


@pytest.mark.parametrize("kind", ["constant", "linear"])
@pytest.mark.parametrize("rv", [None, -73.0, 73.0])
def test_correct_manual_continuum_recovers_gaussian_and_preserves_rv(kind, rv):
    session, correction, c, expected = toy(kind, rv=rv, masked=True)
    original = session.spectrum.flux.copy()
    result = session.edit_line("line1", continuum=correction)
    assert result.status == "converged"
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    assert result.parameters.amplitude == pytest.approx(0.31, abs=1e-9)
    assert result.parameters.centre_air_angstrom == pytest.approx(rest+0.08, abs=1e-9)
    assert result.parameters.sigma_angstrom == pytest.approx(0.34, abs=1e-9)
    assert result.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*0.31*0.34, abs=1e-9)
    working = session.corrected_spectrum("line1")
    region = np.abs(working.wavelength_air_angstrom-rest) <= 3
    np.testing.assert_allclose(working.flux[region], expected[region], rtol=0, atol=1e-12)
    np.testing.assert_allclose(working.uncertainty[region], 0.003, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(working.flux[~region], original[~region])
    np.testing.assert_array_equal(session.spectrum.flux, original)
    np.testing.assert_array_equal(working.mask, session.spectrum.mask)
    np.testing.assert_array_equal(working.wavelength_air_angstrom,
                                  session.spectrum.wavelength_air_angstrom)
    assert working.rv_kms == rv
    assert working.doppler_factor == session.spectrum.doppler_factor
    assert working.flux.flags.writeable is False


def test_repeat_refits_do_not_compound_and_do_not_change_other_lines():
    session, correction, _, _ = toy()
    before = dict(session.fits)
    first = session.edit_line("line1", continuum=correction)
    for _ in range(4):
        assert session.refit_line("line1") == first
    for line in LINE_IDS[1:]:
        assert session.fits[line] is before[line]
    assert session.uncorrected_fits["line1"] == before["line1"]


def test_any_continuum_change_invalidates_approval_and_reset_preserves_masks():
    session, correction, _, _ = toy()
    session.decide("line1", "accepted")
    session.edit_line("line1", continuum=correction)
    assert session.decisions["line1"].state == "pending"
    assert session.reviewed.temperature is None
    session.decide("line1", "accepted", "Manual placement inspected")
    assert session.reviewed.temperature is not None
    rest = session.fits["line1"].rest_wavelength_air_angstrom
    regions = ((rest+1.4, rest+1.7),)
    session.edit_line("line1", excluded_regions=regions)
    session.decide("line1", "accepted")
    session.reset_continuum("line1")
    assert session.decisions["line1"].state == "pending"
    assert session.controls["line1"].excluded_regions == regions
    assert session.controls["line1"].continuum == ContinuumCorrection()
    assert session.corrected_spectrum("line1") is session.spectrum


@pytest.mark.parametrize("kwargs", [
    {"level": 0}, {"level": -1}, {"level": float("nan")}, {"level": float("inf")},
    {"enabled": 1}, {"mode": "automatic"}, {"level": True},
    {"mode": "linear", "anchors": ()},
    {"mode": "linear", "anchors": ((2, 1), (2, 1))},
    {"mode": "linear", "anchors": ((3, 1), (2, 1))},
    {"mode": "linear", "anchors": ((2, 1), (3, 0))},
    {"mode": "linear", "anchors": ((2, 1), (3, float("nan")))},
    {"mode": "linear", "anchors": ((float("inf"), 1), (3, 1))},
    {"mode": "linear", "anchors": ((True, 1), (3, 1))},
])
def test_invalid_manual_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ContinuumCorrection(**kwargs)


def test_positive_anchors_can_still_make_invalid_extrapolation_and_are_rejected_atomically():
    session, _, _, _ = toy()
    session.decide("line1", "accepted")
    before = session.fits["line1"]
    rest = before.rest_wavelength_air_angstrom
    invalid = ContinuumCorrection(
        enabled=True, mode="linear", anchors=((rest+2, 1), (rest+3, 2)),
    )
    with pytest.raises(ValueError, match="strictly positive"):
        session.edit_line("line1", continuum=invalid)
    assert session.fits["line1"] is before
    assert session.decisions["line1"].state == "accepted"


def test_thresholds_are_explicit_configurable_and_warnings_survive_acceptance():
    session, _, _, _ = toy()
    correction = ContinuumCorrection(enabled=True, mode="constant", level=1.15)
    result = session.edit_line("line1", continuum=correction)
    assert "continuum_large_correction" in result.flags
    session.decide("line1", "accepted", "Deliberate analytical example")
    assert "continuum_large_correction" in session.reviewed.fits["line1"].flags
    _, flags = apply_continuum(
        session.spectrum, correction, window=result.window,
        options=ContinuumOptions(max_deviation_warning=0.2),
    )
    assert not flags
    rest = result.rest_wavelength_air_angstrom
    linear = ContinuumCorrection(
        enabled=True, mode="linear", anchors=((rest-1, 1), (rest+1, 1.01)),
    )
    assert "continuum_anchor_extrapolation" in session.edit_line("line1", continuum=linear).flags
    _, flags = apply_continuum(
        session.spectrum, linear, window=result.window,
        options=ContinuumOptions(warn_extrapolation=False),
    )
    assert not flags


@pytest.mark.parametrize("options", [
    {"max_deviation_warning": 0}, {"max_deviation_warning": float("nan")},
    {"max_deviation_warning": -0.1}, {"warn_extrapolation": 1},
])
def test_invalid_warning_thresholds_are_rejected(options):
    with pytest.raises(ValueError):
        ContinuumOptions(**options)


def test_save_restore_recomputes_areas_and_controls_without_arrays(tmp_path):
    session, correction, _, _ = toy("linear")
    session.edit_line("line1", continuum=correction)
    session.decide("line1", "accepted", "Chosen from analytical truth")
    session.notes = "No observed spectra used"
    path = session.save(tmp_path / "review.json")
    payload = read_review(path)
    assert payload["schema_version"] == 2
    assert payload["lines"]["line1"]["controls"]["continuum"] == {
        **asdict(correction), "anchors": [list(a) for a in correction.anchors],
    }
    assert payload["continuum_options"] == asdict(ContinuumOptions())
    assert "corrected_flux" not in path.read_text()
    restored = ReviewSession.load(session.spectrum, path, input_identity=identity(session))
    assert restored.fits["line1"] == session.fits["line1"]
    assert restored.uncorrected_fits["line1"] == session.uncorrected_fits["line1"]
    assert restored.controls == session.controls
    assert restored.decisions == session.decisions
    assert restored.reviewed.temperature == session.reviewed.temperature
    assert restored.notes == session.notes


@pytest.mark.parametrize("change", ["level", "enabled", "options"])
def test_saved_approval_is_bound_to_continuum_even_if_fit_is_numerically_unchanged(change):
    session, _, _, _ = toy("unity")
    session.decide("line1", "accepted")
    payload = session.to_dict()
    if change == "options":
        payload["continuum_options"]["max_deviation_warning"] = 0.2
    else:
        payload["lines"]["line1"]["controls"]["continuum"][change] = (
            1.01 if change == "level" else True
        )
    with pytest.raises(ValueError, match="Accepted controls changed"):
        ReviewSession.from_dict(session.spectrum, payload, input_identity=identity(session))


def test_schema1_approval_checked_before_migrating_to_identity_continuum():
    session, _, _, _ = toy("unity")
    session.decide("line1", "accepted", "PR 3 review")
    legacy = copy.deepcopy(session.to_dict())
    legacy["schema_version"] = 1
    legacy.pop("continuum_options")
    for entry in legacy["lines"].values():
        entry["controls"].pop("continuum")
        if entry["decision"]["state"] == "accepted":
            encoded = json.dumps(
                {"controls": entry["controls"], "fit_options": legacy["fit_options"]},
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            entry["decision"]["controls_fingerprint"] = hashlib.sha256(encoded.encode()).hexdigest()
    restored = ReviewSession.from_dict(session.spectrum, legacy, input_identity=identity(session))
    assert restored.reviewed.temperature == session.reviewed.temperature
    assert restored.fits == session.fits
    assert restored.to_dict()["schema_version"] == 2
    assert all(c.continuum == ContinuumCorrection() for c in restored.controls.values())
    legacy["lines"]["line1"]["controls"]["excluded_regions"] = [[22000, 22001]]
    with pytest.raises(ValueError, match="Accepted controls changed"):
        ReviewSession.from_dict(session.spectrum, legacy, input_identity=identity(session))


def test_manual_evaluation_is_analytical_not_data_derived():
    correction = ContinuumCorrection(
        enabled=True, mode="linear", anchors=((21000, 0.9), (21002, 1.1)),
    )
    np.testing.assert_allclose(evaluate_continuum([20999, 21000, 21001, 21002, 21003], correction),
                               [0.8, 0.9, 1.0, 1.1, 1.2], rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        evaluate_continuum([float("nan")], correction)


@pytest.mark.parametrize("mode", ["constant", "linear"])
def test_explicit_active_unity_correction_reproduces_pr3_results(mode):
    session, _, _, _ = toy("unity")
    baseline = session.fits["line1"]
    correction = ContinuumCorrection(
        enabled=True, mode=mode, level=1,
        anchors=((baseline.window[0], 1), (baseline.window[1], 1)) if mode == "linear" else (),
    )
    assert session.edit_line("line1", continuum=correction) == baseline


def test_reviewed_result_carries_immutable_continuum_metadata_for_standard_outputs():
    session, correction, _, _ = toy("linear")
    assert session.reviewed.continuum_corrections is None
    session.edit_line("line1", continuum=correction)
    reviewed = session.reviewed
    assert reviewed.spectrum is session.spectrum
    assert reviewed.continuum_corrections["line1"] == correction
    assert tuple(reviewed.continuum_corrections) == LINE_IDS
    with pytest.raises(TypeError):
        reviewed.continuum_corrections["line2"] = correction
    session.reset_continuum("line1")
    assert session.reviewed.continuum_corrections is None
