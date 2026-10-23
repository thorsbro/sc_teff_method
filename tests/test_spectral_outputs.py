"""Local outputs and CLI are exercised using generated analytical spectra only."""
# ruff: noqa: E402

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sc_teff_method import FitOptions, fit_spectrum, get_calibration, plot_diagnostics
from sc_teff_method.cli import main
from sc_teff_method.outputs import summary_dict, write_fit_outputs
from sc_teff_method.synthetic import make_synthetic_spectrum

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}
DECLARATIONS = [
    "--wavelength-unit", "angstrom", "--wavelength-frame", "air", "--continuum-normalised",
]


@pytest.fixture(scope="module")
def result():
    synthetic = make_synthetic_spectrum("missing")
    return fit_spectrum(synthetic.wavelength_air_angstrom, synthetic.flux,
                        feh=-0.2, excluded_lines=["line8"], **CONVENTIONS)


@pytest.fixture
def spectrum_file(tmp_path):
    synthetic = make_synthetic_spectrum("noise")
    path = tmp_path / "analytic.csv"
    np.savetxt(path, np.column_stack((synthetic.wavelength_air_angstrom, synthetic.flux,
                                     synthetic.uncertainty, synthetic.mask.astype(int))),
               fmt="%.17g", delimiter=",", comments="",
               header="wavelength,flux,uncertainty,exclude")
    return path


def test_json_csv_and_png_pdf_are_precise_and_preserve_missingness(tmp_path, result):
    output = tmp_path / "results"
    files = write_fit_outputs(result, output, plot_format="both",
                               spectrum_label="Analytical synthetic test")
    assert set(files) == {"summary.json", "lines.csv", "diagnostics.png", "diagnostics.pdf"}
    summary = json.loads(files["summary.json"].read_text())
    assert summary["temperature"]["temperature_k"] == result.temperature.temperature_k
    assert summary["temperature"]["n_lines"] == len(result.included_line_ids)
    assert summary["subset_performance"]["cohort"] == "exact_subset"
    assert summary["calibration"]["calibration_id"] == get_calibration().calibration_id
    assert summary["fits"]["line3"]["area_angstrom"] is None
    assert summary["fits"]["line8"]["status"] == "excluded"
    assert "wavelength_air_angstrom" not in summary["input"]  # no embedded spectral arrays
    with files["lines.csv"].open() as stream:
        rows = {row["line_id"]: row for row in csv.DictReader(stream)}
    assert len(rows) == 9
    assert rows["line3"]["area_angstrom"] == ""
    assert rows["line8"]["line_temperature_k"] == ""
    assert rows["line8"]["included_in_temperature"] == "False"
    assert float(rows["line1"]["area_angstrom"]) == result.fits["line1"].area_angstrom
    assert files["diagnostics.png"].read_bytes().startswith(b"\x89PNG")
    assert files["diagnostics.pdf"].read_bytes().startswith(b"%PDF")
    with pytest.raises(FileExistsError):
        write_fit_outputs(result, output)
    sentinel = output / "unrelated.txt"
    sentinel.write_text("retain this file")
    write_fit_outputs(result, output, overwrite=True, spectrum_label="Analytical synthetic test")
    assert sentinel.read_text() == "retain this file"


def test_all_failed_summary_uses_null_and_plot_has_all_regions(tmp_path):
    synthetic = make_synthetic_spectrum()
    failed = fit_spectrum(synthetic.wavelength_air_angstrom, synthetic.flux,
                           feh=-0.2, options=FitOptions(max_nfev=1), **CONVENTIONS)
    summary = summary_dict(failed)
    assert summary["temperature"] is None
    assert summary["subset_performance"] is None
    assert summary["status"] == "no_usable_lines"
    assert len(summary["exclusion_reasons"]) == 9
    assert "NaN" not in json.dumps(summary, allow_nan=False)
    figure = plot_diagnostics(failed, spectrum_label="Analytical synthetic failure test")
    assert len(figure.axes) == 18
    figure.savefig(tmp_path / "failed.png")
    figure.clear()


def test_partial_output_is_not_published_if_plotting_fails(tmp_path, monkeypatch, result):
    from sc_teff_method import outputs

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(outputs, "plot_diagnostics", fail)
    destination = tmp_path / "failed-output"
    with pytest.raises(OSError):
        write_fit_outputs(result, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".sc-teff-*"))


def test_cli_end_to_end_with_named_columns_and_config(tmp_path, spectrum_file):
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    configuration = tmp_path / "settings.json"
    configuration.write_text(json.dumps({
        "windows": {"line1": [rest-2.5, rest+2.5]},
        "initial_parameters": {"line1": [0.4, rest, 0.4]},
        "options": {"rms_warning": 0.03},
    }))
    output = tmp_path / "cli-results"
    command = [sys.executable, "-m", "sc_teff_method.cli", "fit", str(spectrum_file),
               *DECLARATIONS, "--feh", "-0.2", "--delimiter", ",", "--header",
               "--wavelength-column", "wavelength", "--flux-column", "flux",
               "--uncertainty-column", "uncertainty", "--mask-column", "exclude",
               "--fit-config", str(configuration), "--exclude-line", "line9",
               "--output-dir", str(output), "--label", "Analytical synthetic CLI test"]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert "Teff =" in run.stdout
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["included_line_ids"]) == 8
    assert summary["input"]["sha256"]
    assert summary["input"]["loader_settings"]["mask_column"] == "exclude"
    assert summary["input"]["loader_settings"]["delimiter"] == ","
    assert summary["fits"]["line1"]["window_provisional"] is False
    assert summary["initial_parameter_overrides"]["line1"]["amplitude"] == 0.4
    sentinel = output / "sentinel.txt"
    sentinel.write_text("unchanged")
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 2
    assert "already exists" in again.stderr
    assert sentinel.read_text() == "unchanged"


@pytest.mark.parametrize("extra", [
    [], ["--feh", "nan"], ["--feh", "-0.2", "--initial", "line1:0.4:0.5"],
    ["--feh", "-0.2", "--window", "line10:1:2"],
])
def test_cli_invalid_scientific_or_override_inputs_do_not_write_outputs(
    tmp_path, spectrum_file, extra,
):
    output = tmp_path / "invalid-results"
    code = main(["fit", str(spectrum_file), *DECLARATIONS, "--delimiter", ",", "--header",
                 "--output-dir", str(output), *extra])
    assert code == 2
    assert not output.exists()


def test_cli_demands_explicit_unit_frame_and_continuum_declarations(tmp_path, spectrum_file):
    with pytest.raises(SystemExit) as error:
        main(["fit", str(spectrum_file), "--feh", "-0.2", "--output-dir", str(tmp_path / "out")])
    assert error.value.code == 2


def test_cli_supports_metallicity_free_and_reports_no_usable_lines(tmp_path, spectrum_file):
    code = main(["fit", str(spectrum_file), *DECLARATIONS, "--delimiter", ",", "--header",
                 "--configuration", "nandakumar_no_feh", "--exclude-line", "line1",
                 "--exclude-line", "line2", "--exclude-line", "line3",
                 "--exclude-line", "line4", "--exclude-line", "line5",
                 "--exclude-line", "line6", "--exclude-line", "line7",
                 "--exclude-line", "line8", "--exclude-line", "line9",
                 "--output-dir", str(tmp_path / "none"), "--label", "Analytical synthetic test"])
    assert code == 3
    summary = json.loads((tmp_path / "none/summary.json").read_text())
    assert summary["temperature"] is None
    assert summary["calibration"]["configuration"] == "nandakumar_no_feh"


def test_cli_refuses_to_replace_the_input_spectrum(tmp_path, spectrum_file):
    target = tmp_path / "lines.csv"
    target.write_bytes(spectrum_file.read_bytes())
    original = target.read_bytes()
    assert main(["fit", str(target), *DECLARATIONS, "--feh", "-0.2",
                 "--output-dir", str(tmp_path), "--overwrite"]) == 2
    assert target.read_bytes() == original


def test_cli_expands_tilde_before_protecting_input(tmp_path, spectrum_file, capsys):
    target = tmp_path / "lines.csv"
    target.write_bytes(spectrum_file.read_bytes())
    original = target.read_bytes()
    # Use the real expansion semantics without changing HOME or writing outside tmp_path.
    tilde_directory = Path("~") / os.path.relpath(tmp_path, Path.home())
    code = main(["fit", str(target), *DECLARATIONS, "--delimiter", ",", "--header",
                 "--feh", "-0.2", "--output-dir", str(tilde_directory), "--overwrite"])
    assert code == 2
    assert "replace an input" in capsys.readouterr().err
    assert target.read_bytes() == original


def test_cli_protects_fit_config_from_output_collision(tmp_path, spectrum_file, capsys):
    settings = tmp_path / "summary.json"
    settings.write_text("{}")
    code = main(["fit", str(spectrum_file), *DECLARATIONS, "--delimiter", ",", "--header",
                 "--feh", "-0.2", "--fit-config", str(settings),
                 "--output-dir", str(tmp_path), "--overwrite"])
    assert code == 2
    assert "replace an input" in capsys.readouterr().err
    assert settings.read_text() == "{}"
