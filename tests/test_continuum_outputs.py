"""Exported diagnostics must represent the actual analytically corrected pixels."""
# ruff: noqa: E402

import csv
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sc_teff_method import ContinuumCorrection, estimate_temperature
from sc_teff_method.cli import main
from sc_teff_method.diagnostics import plot_diagnostics
from sc_teff_method.outputs import write_fit_outputs
from sc_teff_method.review import ReviewSession
from sc_teff_method.review_exports import write_review_outputs
from sc_teff_method.review_io import load_spectrum_identity
from sc_teff_method.synthetic import make_continuum_demo

SETTINGS = {"delimiter": ",", "header": True, "wavelength_column": "wave",
            "flux_column": "flux", "mask_column": "exclude"}


@pytest.fixture
def local_review(tmp_path):
    toy = make_continuum_demo()
    source = tmp_path / "invented.csv"
    np.savetxt(source, np.column_stack((toy.wavelength_air_angstrom, toy.flux, toy.mask)),
               delimiter=",", header="wave,flux,exclude", comments="", fmt="%.17g")
    spectrum, identity = load_spectrum_identity(
        source, loader_settings=SETTINGS, wavelength_unit="angstrom", wavelength_frame="air",
        continuum_normalised=True,
    )
    session = ReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    for line in ("line2", "line3"):
        prescription = toy.continuum_truth[line]
        session.edit_line(line, continuum=ContinuumCorrection(
            enabled=True, mode=prescription.mode, level=prescription.level,
            anchors=prescription.anchors,
        ))
        session.decide(line, "accepted", "Inspected a prescribed mathematical correction.")
    return source, session


def test_summary_csv_and_review_record_continuum_without_spectral_arrays(local_review, tmp_path):
    source, session = local_review
    files = write_review_outputs(session, tmp_path / "final", protected_paths=[source])
    summary = json.loads(files["summary.json"].read_text())
    review = json.loads(files["review.json"].read_text())
    assert summary["schema_version"] == review["schema_version"] == 2
    assert summary["continuum_options"] == review["continuum_options"]
    assert "placement" in summary["continuum_note"]
    assert "corrections" in summary["review"]["selection_note"]
    assert summary["included_line_ids"] == ["line2", "line3"]
    assert summary["fits"]["line2"]["continuum"]["level"] == 1.035
    assert summary["fits"]["line3"]["continuum"]["mode"] == "linear"
    for line in ("line2", "line3"):
        fit = summary["fits"][line]
        assert fit["review_state"] == "accepted"
        assert fit["area_angstrom"] == session.fits[line].area_angstrom
        assert fit["area_error_angstrom"] == session.fits[line].area_error_angstrom
        assert fit["uncorrected_area_angstrom"] == session.uncorrected_fits[line].area_angstrom
        assert fit["uncorrected_area_angstrom"] != fit["area_angstrom"]
        expected = estimate_temperature(
            {line: fit["uncorrected_area_angstrom"]}, feh=-0.2,
        ).temperature_k
        assert fit["uncorrected_line_temperature_k"] == expected
    with files["lines.csv"].open() as stream:
        rows = {row["line_id"]: row for row in csv.DictReader(stream)}
    assert len(rows) == 9
    assert rows["line1"]["review_state"] == "pending"
    assert rows["line2"]["continuum_enabled"] == "True"
    assert rows["line3"]["continuum_mode"] == "linear"
    assert json.loads(rows["line3"]["continuum_anchors"]) == (
        review["lines"]["line3"]["controls"]["continuum"]["anchors"]
    )
    for payload in (review, summary):
        encoded = json.dumps(payload)
        assert '"corrected_flux"' not in encoded
        assert '"wavelength_air_angstrom": [' not in encoded
        assert '"flux": [' not in encoded


@pytest.mark.parametrize("explicit", [False, True])
def test_diagnostic_curves_use_corrected_flux_and_residuals(local_review, explicit):
    _, session = local_review
    result = session.reviewed
    line_spectra = {line: session.corrected_spectrum(line) for line in session.fits}
    controls = {line: control.continuum for line, control in session.controls.items()}
    figure = (plot_diagnostics(result, line_spectra=line_spectra, continuum_corrections=controls)
              if explicit else plot_diagnostics(result))
    try:
        profile, residual = figure.axes[2:4]  # line2 constant correction
        x = np.array(profile.lines[1].get_xdata())
        plotted = np.array(profile.lines[1].get_ydata())
        indices = np.searchsorted(session.spectrum.wavelength_air_angstrom, x)
        np.testing.assert_array_equal(plotted, session.spectrum.flux[indices] / 1.035)
        assert not np.array_equal(plotted, session.spectrum.flux[indices])
        assert np.max(np.abs(residual.lines[0].get_ydata())) < 1e-10
        curve = next(line for line in profile.lines if line.get_label() == "Manual continuum C")
        np.testing.assert_array_equal(curve.get_ydata(), np.full(100, 1.035))
        original = next(line for line in profile.lines if line.get_label() == "Original input")
        indices = np.searchsorted(
            session.spectrum.wavelength_air_angstrom, original.get_xdata(),
        )
        np.testing.assert_array_equal(original.get_ydata(), session.spectrum.flux[indices])
    finally:
        figure.clear()


def test_existing_fit_output_function_preserves_reviewed_continuum_metadata(local_review, tmp_path):
    _, session = local_review
    # Generic PR 2 output must also plot the corrected pixels and report C.
    files = write_fit_outputs(session.reviewed, tmp_path / "generic-fit-output")
    summary = json.loads(files["summary.json"].read_text())
    assert summary["schema_version"] == 2
    assert summary["fits"]["line2"]["continuum"]["level"] == 1.035
    assert summary["fits"]["line3"]["continuum"]["mode"] == "linear"
    assert summary["temperature"]["temperature_k"] == session.reviewed.temperature.temperature_k
    assert "continuum-placement" in summary["continuum_note"]
    with files["lines.csv"].open() as stream:
        rows = {row["line_id"]: row for row in csv.DictReader(stream)}
    assert rows["line2"]["continuum_mode"] == "constant"
    assert rows["line3"]["continuum_enabled"] == "True"


def test_inspect_continuum_thresholds_saved_and_restore_rejects_overrides(
    local_review, tmp_path, monkeypatch, capsys,
):
    pytest.importorskip("bokeh")
    from sc_teff_method import inspection

    source, original = local_review
    captured = []
    monkeypatch.setattr(
        inspection, "serve_inspection", lambda session, **kw: captured.append(session),
    )
    assert main([
        "inspect", str(source), "--delimiter", ",", "--header", "--mask-column", "exclude",
        "--wavelength-column", "wave", "--flux-column", "flux",
        "--wavelength-unit", "angstrom", "--wavelength-frame", "air",
        "--continuum-normalised", "--feh", "-0.2", "--continuum-warning-deviation", "0.03",
        "--no-continuum-extrapolation-warning",
    ]) == 0
    assert captured[0].continuum_options.max_deviation_warning == 0.03
    assert captured[0].continuum_options.warn_extrapolation is False
    review = original.save(tmp_path / "saved.json")
    assert main([
        "inspect", str(source), "--review-file", str(review),
        "--continuum-warning-deviation", "0.2",
    ]) == 2
    assert "saved controls" in capsys.readouterr().err
    assert len(captured) == 1


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_invalid_cli_threshold_never_launches(local_review, monkeypatch, value):
    pytest.importorskip("bokeh")
    from sc_teff_method import inspection

    source, _ = local_review
    monkeypatch.setattr(inspection, "serve_inspection", lambda *a, **k: pytest.fail("launched"))
    assert main([
        "inspect", str(source), "--delimiter", ",", "--header", "--mask-column", "exclude",
        "--wavelength-column", "wave", "--flux-column", "flux", "--wavelength-unit", "angstrom",
        "--wavelength-frame", "air", "--continuum-normalised", "--feh", "-0.2",
        f"--continuum-warning-deviation={value}",
    ]) == 2
