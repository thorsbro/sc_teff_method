"""Scientific review and persistence checks use mathematical spectra only."""
# ruff: noqa: E402

import copy
import hashlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sc_teff_method import LINE_IDS, estimate_temperature, fitting, subset_performance
from sc_teff_method.fitting import FitOptions, GaussianParameters, fit_gaussian_lines
from sc_teff_method.review import ReviewSession, read_review
from sc_teff_method.spectrum import prepare_spectrum
from sc_teff_method.synthetic import make_synthetic_spectrum

CONVENTIONS = {
    "wavelength_unit": "angstrom",
    "wavelength_frame": "air",
    "continuum_normalised": True,
}


def session(case="clean", **kwargs):
    toy = make_synthetic_spectrum(case)
    prepared = prepare_spectrum(
        toy.wavelength_air_angstrom,
        toy.flux,
        uncertainty=toy.uncertainty,
        mask=toy.mask,
        rv_kms=toy.rv_kms,
        **CONVENTIONS,
    )
    identity = {
        "filename": "analytical.csv",
        "sha256": hashlib.sha256(
            toy.wavelength_air_angstrom.tobytes() + toy.flux.tobytes()
        ).hexdigest(),
        "loader_settings": {},
    }
    return ReviewSession.create(prepared, input_identity=identity, feh=-0.2, **kwargs)


def identity_of(review):
    return {
        **review.input_identity,
        "loader_settings": dict(review.input_identity["loader_settings"]),
    }


@pytest.mark.parametrize("configuration", ["nandakumar_feh", "nandakumar_no_feh"])
def test_manual_acceptance_required_and_uses_unchanged_calibration(configuration):
    review = session(configuration=configuration)
    assert review.reviewed.temperature is None
    assert review.reviewed.performance is None
    assert all(decision.state == "pending" for decision in review.decisions.values())
    review.decide("line1", "accepted", "Inspected analytic line")
    review.decide("line2", "accepted")
    accepted = {line: review.fits[line].area_angstrom for line in ("line1", "line2")}
    expected = estimate_temperature(accepted, configuration=configuration, feh=-0.2)
    assert review.reviewed.temperature == expected
    assert review.reviewed.performance == subset_performance(accepted, configuration=configuration)
    assert review.reviewed.included_line_ids == ("line1", "line2")
    assert any("post-hoc" in warning for warning in review.reviewed.warnings)
    review.decide("line1", "rejected", "Deliberate exclusion")
    assert review.reviewed.included_line_ids == ("line2",)
    review.decide("line2", "pending")
    assert review.reviewed.temperature is None


def test_quality_flags_require_a_manual_decision_and_remain_visible():
    review = session("blends")
    flags = review.fits["line4"].flags
    assert flags and review.reviewed.temperature is None
    review.decide("line4", "accepted", "Synthetic blend inspected deliberately")
    assert review.reviewed.included_line_ids == ("line4",)
    assert review.fits["line4"].flags == flags
    assert review.decisions["line4"].controls_fingerprint
    assert any("explicitly manually accepted" in warning for warning in review.reviewed.warnings)


@pytest.mark.parametrize(
    "case,line,kwargs",
    [
        ("missing", "line3", {}),
        ("clean", "line1", {"options": FitOptions(max_nfev=1)}),
        ("clean", "line1", {"excluded_lines": ["line1"]}),
    ],
)
def test_unavailable_failed_and_excluded_measurements_cannot_be_accepted(case, line, kwargs):
    review = session(case, **kwargs)
    assert review.fits[line].area_angstrom is None
    with pytest.raises(ValueError, match="valid converged"):
        review.decide(line, "accepted")
    assert review.reviewed.temperature is None


def test_one_line_edit_uses_existing_fitter_and_preserves_eight_other_fits(monkeypatch):
    review = session("blends")
    review.decide("line1", "accepted", "retain this decision")
    review.decide("line4", "accepted")
    before = dict(review.fits)
    calls = []
    original = fitting.least_squares

    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, "least_squares", track)
    regions = ((22024.6, 22025.3),)
    changed = review.edit_line("line4", excluded_regions=regions)
    assert len(calls) == 1
    assert review.decisions["line4"].state == "pending"
    assert review.decisions["line4"].controls_fingerprint is None
    assert review.decisions["line1"].state == "accepted"
    for line in LINE_IDS:
        if line != "line4":
            assert review.fits[line] is before[line]
    direct = fit_gaussian_lines(review.spectrum, excluded_regions={"line4": regions})["line4"]
    assert changed == direct
    assert changed.area_angstrom != pytest.approx(before["line4"].area_angstrom)


@pytest.mark.parametrize("change", ["window", "initial", "refit", "reset"])
def test_fitting_control_changes_invalidate_scientific_acceptance(change):
    review = session()
    review.decide("line1", "accepted", "Old note")
    rest = review.fits["line1"].rest_wavelength_air_angstrom
    if change == "window":
        review.edit_line("line1", window=(rest - 2.5, rest + 2.5))
    elif change == "initial":
        review.edit_line("line1", initial=GaussianParameters(0.4, rest, 0.45))
    elif change == "refit":
        review.refit_line("line1")
    else:
        review.reset_line("line1")
    assert review.decisions["line1"].state == "pending"
    assert review.decisions["line1"].note == ""
    assert review.reviewed.temperature is None


def test_invalid_control_does_not_replace_a_valid_committed_fit():
    review = session()
    review.decide("line1", "accepted")
    fit = review.fits["line1"]
    with pytest.raises(ValueError):
        review.edit_line("line1", window=(3, 2))
    assert review.fits["line1"] is fit
    assert review.decisions["line1"].state == "accepted"


def test_reset_removes_local_edits_but_retains_original_spectrum_mask_and_rv():
    review = session("rv_offset")
    original_wave = review.spectrum.wavelength_air_angstrom.copy()
    original_flux = review.spectrum.flux.copy()
    rest = review.fits["line1"].rest_wavelength_air_angstrom
    review.edit_line(
        "line1",
        window=(rest - 2, rest + 2),
        initial=GaussianParameters(0.4, rest, 0.45),
        excluded_regions=((rest + 1, rest + 1.5),),
    )
    review.reset_line("line1")
    assert review.controls["line1"].window is None
    assert review.controls["line1"].initial is None
    assert review.controls["line1"].excluded_regions == ()
    assert review.fits["line1"].window_provisional
    np.testing.assert_array_equal(review.spectrum.wavelength_air_angstrom, original_wave)
    np.testing.assert_array_equal(review.spectrum.flux, original_flux)
    assert review.spectrum.rv_kms == 65


def test_review_round_trip_has_no_spectrum_arrays_or_absolute_source_path(tmp_path):
    review = session("blends")
    review.edit_line("line4", excluded_regions=((22024.6, 22025.3),))
    review.decide("line1", "accepted", "Clean")
    review.decide("line4", "accepted", "Masked synthetic neighbour")
    review.decide("line6", "rejected", "Wrong synthetic continuum")
    review.notes = "Mathematical demonstration only"
    path = review.save(tmp_path / "review.json")
    payload = read_review(path)
    assert payload["input"]["filename"] == "analytical.csv"
    assert "flux" not in payload["input"]
    assert "wavelength_air_angstrom" not in payload["input"]
    assert len(path.read_text()) < 25000
    restored = ReviewSession.load(review.spectrum, path, input_identity=identity_of(review))
    assert restored.reviewed.temperature == review.reviewed.temperature
    assert restored.reviewed.performance == review.reviewed.performance
    assert dict(restored.decisions) == dict(review.decisions)
    assert dict(restored.controls) == dict(review.controls)
    assert restored.notes == review.notes
    for line in LINE_IDS:
        assert restored.fits[line] == review.fits[line]


@pytest.mark.parametrize("change", ["checksum", "parser", "rv", "prepared"])
def test_stale_or_differently_prepared_spectrum_is_rejected(change):
    review = session()
    payload = review.to_dict()
    identity = identity_of(review)
    if change == "checksum":
        identity["sha256"] = "0" * 64
    elif change == "parser":
        identity["loader_settings"]["flux_column"] = 2
    elif change == "rv":
        payload["input"]["rv_kms"] = 1
    else:
        payload["input"]["prepared_spectrum_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        ReviewSession.from_dict(review.spectrum, payload, input_identity=identity)


def test_moving_and_renaming_identical_input_is_portable():
    review = session()
    identity = identity_of(review)
    identity["filename"] = "renamed.csv"
    restored = ReviewSession.from_dict(review.spectrum, review.to_dict(), input_identity=identity)
    assert restored.input_identity["filename"] == "renamed.csv"


@pytest.mark.parametrize("change", ["schema", "calibration", "line", "snapshot", "unknown"])
def test_malformed_or_unknown_review_records_are_rejected(change):
    review = session()
    payload = review.to_dict()
    if change == "schema":
        payload["schema_version"] = 2
    elif change == "calibration":
        payload["calibration"]["calibration_id"] = "other"
    elif change == "line":
        payload["lines"]["line10"] = payload["lines"].pop("line9")
    elif change == "snapshot":
        payload["lines"]["line1"]["fit"]["area_angstrom"] += 0.1
    else:
        payload["spectrum"] = {"flux": [1, 1]}
    with pytest.raises(ValueError):
        ReviewSession.from_dict(review.spectrum, payload, input_identity=identity_of(review))


def test_changing_saved_accepted_controls_requires_reapproval_even_when_fit_is_identical():
    review = session()
    review.decide("line1", "accepted")
    payload = review.to_dict()
    rest = review.fits["line1"].rest_wavelength_air_angstrom
    payload["lines"]["line1"]["controls"]["initial"] = {
        "amplitude": 0.4,
        "centre_air_angstrom": rest,
        "sigma_angstrom": 0.45,
    }
    with pytest.raises(ValueError, match="Accepted controls changed"):
        ReviewSession.from_dict(review.spectrum, payload, input_identity=identity_of(review))


def test_saved_snapshots_never_supply_recalculated_areas(monkeypatch):
    review = session()
    review.decide("line1", "accepted")
    payload = review.to_dict()
    original = fitting.least_squares

    def perturbed(*args, **kwargs):
        result = original(*args, **kwargs)
        result.x[0] += 0.001
        return result

    monkeypatch.setattr(fitting, "least_squares", perturbed)
    with pytest.raises(ValueError, match="Accepted fit changed"):
        ReviewSession.from_dict(review.spectrum, payload, input_identity=identity_of(review))


def test_save_overwrite_is_explicit_and_never_replaces_source_or_config(tmp_path):
    review = session()
    path = review.save(tmp_path / "review.json")
    with pytest.raises(FileExistsError):
        review.save(path)
    review.notes = "explicit replacement"
    review.save(path, overwrite=True)
    assert read_review(path)["notes"] == review.notes
    source = tmp_path / "source.csv"
    source.write_text("21730,0.8\n21731,1\n")  # invented, never observed
    with pytest.raises(ValueError, match="protected"):
        review.save(source, overwrite=True, protected_paths=[source])
    assert source.read_text().startswith("21730,0.8")
    payload = source.read_bytes()
    review._identity["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="source spectrum"):
        review.save(source, overwrite=True)
    assert source.read_bytes() == payload


@pytest.mark.parametrize(
    "content",
    [
        '{"format":"x","format":"y"}',
        '{"format":NaN}',
        '{"format":Infinity}',
    ],
)
def test_duplicate_or_nonfinite_json_is_rejected(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_review(path)


def test_public_control_and_decision_maps_are_read_only():
    review = session()
    with pytest.raises(TypeError):
        review.decisions["line1"] = None
    with pytest.raises(TypeError):
        review.controls["line1"] = None
    with pytest.raises(TypeError):
        review.input_identity["loader_settings"]["flux_column"] = 7
    original = copy.deepcopy(review.to_dict())
    review.notes = "new note"
    assert original["notes"] == ""
