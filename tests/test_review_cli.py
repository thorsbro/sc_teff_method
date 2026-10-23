"""Portable review/replay and exports use only analytical synthetic spectra."""
# ruff: noqa: E402

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sc_teff_method import CONFIGURATIONS, estimate_temperature
from sc_teff_method.cli import main
from sc_teff_method.review import ReviewSession
from sc_teff_method.review_exports import write_review_outputs
from sc_teff_method.review_io import load_spectrum_identity
from sc_teff_method.synthetic import make_review_demo

SETTINGS = {
    "wavelength_column": "wave",
    "flux_column": "flux",
    "uncertainty_column": "error",
    "mask_column": "exclude",
    "delimiter": ",",
    "header": True,
    "skiprows": 0,
}
CONVENTIONS = {
    "wavelength_unit": "angstrom",
    "wavelength_frame": "air",
    "continuum_normalised": True,
}


@pytest.fixture
def source(tmp_path):
    toy = make_review_demo()
    path = tmp_path / "analytical.csv"
    np.savetxt(
        path,
        np.column_stack(
            (toy.wavelength_air_angstrom, toy.flux, toy.uncertainty, toy.mask.astype(int))
        ),
        delimiter=",",
        header="wave,flux,error,exclude",
        comments="",
        fmt="%.17g",
    )
    return path


def session_for(path, configuration="nandakumar_feh"):
    spectrum, identity = load_spectrum_identity(path, loader_settings=SETTINGS, **CONVENTIONS)
    return ReviewSession.create(
        spectrum,
        input_identity=identity,
        configuration=configuration,
        feh=-0.2 if configuration == "nandakumar_feh" else None,
    )


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_review_exports_have_only_accepted_temperature_and_all_nine_states(
    source,
    tmp_path,
    configuration,
):
    session = session_for(source, configuration)
    session.decide("line1", "accepted", note="Inspected the mathematical clean line.")
    session.decide("line4", "accepted", note="Deliberate flagged synthetic comparison.")
    session.decide("line6", "rejected", note="Invented continuum error.")
    session.notes = "Analytical test; no observed spectrum."
    target = tmp_path / "products"
    files = write_review_outputs(
        session,
        target,
        plot_format="both",
        spectrum_label="Analytical review test",
        protected_paths=[source],
    )
    summary = json.loads(files["summary.json"].read_text())
    assert summary["estimate_kind"] == "explicitly_reviewed"
    assert summary["included_line_ids"] == ["line1", "line4"]
    expected = estimate_temperature(
        {line: session.fits[line].area_angstrom for line in ("line1", "line4")},
        configuration=configuration,
        feh=session.feh,
    )
    assert summary["temperature"]["temperature_k"] == expected.temperature_k
    assert summary["subset_performance"]["cohort"] == "exact_subset"
    assert summary["fits"]["line6"]["review_state"] == "rejected"
    assert "post-hoc" in summary["review"]["selection_note"]
    review = json.loads(files["review.json"].read_text())
    assert len(review["lines"]) == 9
    assert review["input"]["filename"] == source.name
    assert "wavelength_air_angstrom" not in review["input"]
    assert not Path(review["input"]["filename"]).is_absolute()
    with files["lines.csv"].open() as stream:
        rows = {row["line_id"]: row for row in csv.DictReader(stream)}
    assert rows["line1"]["review_state"] == "accepted"
    assert rows["line4"]["accepted_with_quality_flags"] == "True"
    assert rows["line6"]["review_state"] == "rejected"
    assert rows["line5"]["review_state"] == "pending"
    assert rows["line5"]["included_in_temperature"] == "False"
    assert files["diagnostics.png"].read_bytes().startswith(b"\x89PNG")
    assert files["diagnostics.pdf"].read_bytes().startswith(b"%PDF")


def test_replay_uses_saved_parser_and_decisions_on_a_renamed_portable_input(source, tmp_path):
    session = session_for(source)
    session.decide("line1", "accepted")
    session.decide("line5", "accepted")
    review = tmp_path / "saved.json"
    session.save(review)
    moved = tmp_path / "moved"
    moved.mkdir()
    renamed = moved / "different-name.csv"
    shutil.copyfile(source, renamed)
    command = [
        sys.executable,
        "-m",
        "sc_teff_method.cli",
        "replay",
        str(renamed),
        "--review-file",
        str(review),
        "--output-dir",
        str(moved / "final"),
        "--label",
        "Analytical replay",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Reviewed Teff" in result.stdout
    summary = json.loads((moved / "final/summary.json").read_text())
    assert summary["temperature"]["temperature_k"] == session.reviewed.temperature.temperature_k
    assert summary["included_line_ids"] == ["line1", "line5"]
    assert summary["input"]["filename"] == renamed.name
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 2
    assert "already exists" in again.stderr


def test_no_approval_replay_keeps_final_temperature_missing(source, tmp_path):
    session = session_for(source)
    review = tmp_path / "pending.json"
    session.save(review)
    output = tmp_path / "pending-output"
    assert (
        main(
            [
                "replay",
                str(source),
                "--review-file",
                str(review),
                "--output-dir",
                str(output),
                "--label",
                "Analytical pending test",
            ]
        )
        == 3
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["temperature"] is None
    assert summary["subset_performance"] is None
    assert summary["included_line_ids"] == []


def test_stale_checksum_rejected_even_if_parsed_flux_would_be_identical(source, tmp_path, capsys):
    session = session_for(source)
    session.decide("line1", "accepted")
    review = tmp_path / "review.json"
    session.save(review)
    source.write_bytes(source.read_bytes() + b"# artificial checksum-only change\n")
    output = tmp_path / "stale"
    assert (
        main(["replay", str(source), "--review-file", str(review), "--output-dir", str(output)])
        == 2
    )
    assert "checksum" in capsys.readouterr().err
    assert not output.exists()


def test_review_replay_cannot_overwrite_source_or_input_review(source, tmp_path):
    target_source = tmp_path / "lines.csv"
    shutil.copyfile(source, target_source)
    session = session_for(target_source)
    session.decide("line1", "accepted")
    review = tmp_path / "review.json"
    session.save(review)
    original_source, original_review = target_source.read_bytes(), review.read_bytes()
    assert (
        main(
            [
                "replay",
                str(target_source),
                "--review-file",
                str(review),
                "--output-dir",
                str(tmp_path),
                "--overwrite",
            ]
        )
        == 2
    )
    assert target_source.read_bytes() == original_source
    assert review.read_bytes() == original_review


def test_export_failure_leaves_no_partial_directory(source, tmp_path, monkeypatch):
    import sc_teff_method.review_exports as exports

    session = session_for(source)
    session.decide("line1", "accepted")

    def fail(*args, **kwargs):
        raise OSError("synthetic export failure")

    monkeypatch.setattr(exports, "plot_diagnostics", fail)
    with pytest.raises(OSError):
        write_review_outputs(session, tmp_path / "failed", protected_paths=[source])
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".sc-teff-review-*"))


def test_inspect_cli_restores_without_extra_input_conventions_and_rejects_overrides(
    source,
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("bokeh")
    import sc_teff_method.inspection as inspection

    session = session_for(source)
    session.decide("line1", "accepted")
    review = tmp_path / "review.json"
    session.save(review)
    launched = []
    monkeypatch.setattr(
        inspection, "serve_inspection", lambda restored, **kw: launched.append(restored)
    )
    assert main(["inspect", str(source), "--review-file", str(review), "--no-browser"]) == 0
    assert launched[0].reviewed.temperature == session.reviewed.temperature
    assert main(["inspect", str(source), "--review-file", str(review), "--rv-kms", "0"]) == 2
    assert len(launched) == 1


def test_new_inspection_requires_explicit_conventions(source):
    pytest.importorskip("bokeh")
    assert main(["inspect", str(source), "--feh", "-0.2", "--no-browser"]) == 2
