"""HFS method-specific review tests use only analytical synthetic HFS profiles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sc_teff_method import estimate_hfs_temperature
from sc_teff_method.continuum import ContinuumCorrection
from sc_teff_method.hfs_review import HFSReviewSession, read_hfs_review
from sc_teff_method.hfs_review_exports import write_hfs_review_outputs
from sc_teff_method.hfs_synthetic import make_hfs_synthetic_spectrum
from sc_teff_method.review import ReviewSession
from sc_teff_method.review_io import load_spectrum_identity
from sc_teff_method.spectrum import prepare_spectrum


def make_sessions(kind="clean"):
    toy = make_hfs_synthetic_spectrum(kind)
    spectrum = prepare_spectrum(
        toy.wavelength_air_angstrom, toy.flux, uncertainty=toy.uncertainty,
        mask=toy.mask, rv_kms=toy.rv_kms,
        wavelength_unit="angstrom", wavelength_frame="air",
        continuum_normalised=True,
    )
    identity = {
        "filename": "entirely-analytical.csv",
        "sha256": hashlib.sha256(
            toy.wavelength_air_angstrom.tobytes() + toy.flux.tobytes()
        ).hexdigest(),
        "loader_settings": {},
    }
    hfs = HFSReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    gaussian = ReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    return toy, hfs, gaussian


def test_hfs_requires_independent_explicit_scientific_acceptance():
    toy, hfs, gaussian = make_sessions()
    assert hfs.reviewed.temperature is None
    assert all(d.state == "pending" for d in hfs.decisions.values())
    gaussian.decide("line1", "accepted")
    assert hfs.reviewed.temperature is None
    hfs.decide("line1", "accepted", "Inspected analytical HFS profile")
    assert hfs.reviewed.included_line_ids == ("line1",)
    expected = estimate_hfs_temperature(
        {"line1": hfs.fits["line1"].area_angstrom}, feh=-0.2,
    )
    assert hfs.reviewed.temperature.temperature_k == expected.temperature_k
    assert hfs.reviewed.strict_strength_domain_applicable is None
    assert gaussian.decisions["line1"].state == "accepted"
    assert gaussian.reviewed.temperature is not None
    assert abs(hfs.fits["line1"].area_angstrom
               - gaussian.fits["line1"].area_angstrom) > 1e-4


def test_edit_mask_window_and_continuum_invalidate_only_hfs_acceptance():
    _, hfs, gaussian = make_sessions()
    hfs.decide("line1", "accepted")
    gaussian.decide("line1", "accepted")
    original = dict(hfs.fits)
    rest = original["line1"].rest_wavelength_air_angstrom
    fitted = hfs.edit_line(
        "line1", window=(rest-2.5, rest+2.5),
        excluded_regions=((rest+0.7, rest+0.9),),
        continuum=ContinuumCorrection(enabled=True, mode="constant", level=1.01),
    )
    assert hfs.decisions["line1"].state == "pending"
    assert gaussian.decisions["line1"].state == "accepted"
    assert hfs.reviewed.temperature is None
    assert fitted is hfs.fits["line1"]
    assert all(hfs.fits[line] is original[line] for line in hfs.fits if line != "line1")
    hfs.decide("line1", "accepted")
    hfs.refit_line("line1")
    assert hfs.decisions["line1"].state == "pending"
    hfs.reset_continuum("line1")
    assert not hfs.controls["line1"].continuum.active
    assert hfs.controls["line1"].window is not None


def test_hfs_review_is_portable_and_recomputed(tmp_path):
    toy, session, gaussian = make_sessions()
    session.decide("line1", "accepted", "Synthetic HFS fit")
    session.decide("line4", "rejected", "Invented feature")
    session.notes = "Independent HFS reviewer"
    before = session.reviewed.temperature.temperature_k
    p = session.save(tmp_path / "hfs-review.json")
    payload = read_hfs_review(p)
    assert payload["measurement_source"] == "hfs"
    assert payload["atomic_dataset_id"] == "hala2022_ab_generated"
    assert "wavelength_air_angstrom" not in p.read_text()
    assert len(p.read_bytes()) < 100_000
    restored = HFSReviewSession.load(session.spectrum, p, input_identity=session.input_identity)
    assert restored.reviewed.temperature.temperature_k == pytest.approx(before, abs=1e-9)
    assert restored.decisions["line1"].state == "accepted"
    assert restored.decisions["line4"].state == "rejected"
    assert restored.notes == session.notes

    bad = dict(session.input_identity)
    bad["sha256"] = "0"*64
    with pytest.raises(ValueError, match="checksum"):
        HFSReviewSession.from_dict(session.spectrum, payload, input_identity=bad)
    payload["lines"]["line1"]["fit"]["area_angstrom"] += 0.02
    with pytest.raises(ValueError, match="Accepted HFS fit"):
        HFSReviewSession.from_dict(session.spectrum, payload,
                                   input_identity=session.input_identity)


def test_hfs_review_rejects_stale_acceptance_after_continuum_edit():
    _, session, _ = make_sessions()
    session.decide("line1", "accepted")
    payload = session.to_dict()
    payload["lines"]["line1"]["controls"]["continuum"] = {
        "enabled": True, "mode": "constant", "level": 1.04, "anchors": [],
    }
    with pytest.raises(ValueError, match="Accepted HFS controls"):
        HFSReviewSession.from_dict(session.spectrum, payload,
                                   input_identity=session.input_identity)


def test_reviewed_hfs_export_is_hfs_only(tmp_path):
    _, session, _ = make_sessions()
    session.decide("line1", "accepted", "Synthetic area")
    target = tmp_path / "hfs-reviewed"
    written = write_hfs_review_outputs(session, target)
    assert {"summary.json", "lines.csv", "diagnostics.png", "review.json"} <= set(written)
    summary = json.loads((target / "summary.json").read_text())
    assert summary["format"] == "sc-teff-hfs-reviewed"
    assert summary["measurement_source"] == "hfs"
    assert summary["included_line_ids"] == ["line1"]
    assert summary["review"]["decisions"]["line1"]["state"] == "accepted"
    with pytest.raises(FileExistsError):
        write_hfs_review_outputs(session, target)


def test_saved_hfs_review_replays_without_bokeh(tmp_path):
    from sc_teff_method.cli import main

    toy = make_hfs_synthetic_spectrum("clean")
    path = tmp_path / "analytic-hfs.csv"
    original = np.column_stack((toy.wavelength_air_angstrom, toy.flux))
    np.savetxt(path, original, delimiter=",", header="wave,flux", comments="", fmt="%.17g")
    input_spectrum, identity = load_spectrum_identity(
        path, loader_settings={"wavelength_column": "wave", "flux_column": "flux",
                               "header": True, "delimiter": ","},
        wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
    )
    session = HFSReviewSession.create(input_spectrum, input_identity=identity, feh=-0.2)
    session.decide("line1", "accepted")
    session.decide("line2", "accepted")
    saved = session.save(tmp_path / "review-hfs.json")
    output = tmp_path / "reproduced"
    status = main(["replay-hfs", str(path), "--review-file", str(saved),
                   "--output-dir", str(output)])
    assert status == 0
    result = json.loads((output / "summary.json").read_text())
    assert result["temperature"]["temperature_k"] == pytest.approx(
        session.reviewed.temperature.temperature_k, abs=1e-9,
    )
    assert (output / "review.json").exists()
    assert np.array_equal(np.loadtxt(path, delimiter=",", skiprows=1), original)
