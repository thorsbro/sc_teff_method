"""HFS workflow integration uses analytical profiles only."""
# ruff: noqa: E402
import csv
import json
import subprocess
import sys
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sc_teff_method import (
    ContinuumCorrection,
    HFSFitOptions,
    LineAreas,
    estimate_hfs_temperature,
    estimate_temperature,
    fit_hfs_spectrum,
)
from sc_teff_method.cli import main
from sc_teff_method.hfs_outputs import hfs_summary_dict, plot_hfs_diagnostics, write_hfs_outputs
from sc_teff_method.hfs_synthetic import make_hfs_synthetic_spectrum
from sc_teff_method.hfs_workflow import estimate_hfs_from_spectrum
from sc_teff_method.outputs import summary_dict, write_fit_outputs

DECLARATIONS = dict(wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True)
CLI_DECLARATIONS = ["--wavelength-unit", "angstrom", "--wavelength-frame", "air",
                    "--continuum-normalised"]


@pytest.fixture(scope="module")
def result():
    toy = make_hfs_synthetic_spectrum("noise")
    return fit_hfs_spectrum(toy.wavelength_air_angstrom, toy.flux, uncertainty=toy.uncertainty,
                            mask=toy.mask, feh=-0.2, **DECLARATIONS)


@pytest.mark.parametrize("configuration", ["nandakumar_feh", "nandakumar_no_feh"])
def test_hfs_workflow_uses_own_coefficients_and_mean_of_line_temperatures(configuration):
    toy = make_hfs_synthetic_spectrum()
    result = fit_hfs_spectrum(
        toy.wavelength_air_angstrom, toy.flux, **DECLARATIONS, configuration=configuration,
        feh=-0.2 if configuration == "nandakumar_feh" else None,
    )
    assert len(result.included_line_ids) == 9
    expected = estimate_hfs_temperature(result.areas, configuration=configuration, feh=result.feh)
    assert result.temperature == expected
    assert result.temperature.measurement_source == result.performance.measurement_source == "hfs"
    assert result.temperature.temperature_k == pytest.approx(
        sum(result.line_temperatures_k.values())/9, abs=1e-12,
    )
    gaussian = estimate_temperature(dict(result.areas), configuration=configuration, feh=result.feh)
    assert abs(gaussian.temperature_k-result.temperature.temperature_k) > 1
    assert result.strict_strength_domain_applicable is None
    assert any(w.code == "strict_strength_domain_unknown" for w in result.calibration_warnings)
    assert any("does not establish target applicability" in w for w in result.warnings)


@pytest.mark.parametrize("source", ["gaussian", "hfs"])
def test_method_tag_prevents_accidental_calibration_mixing(source):
    areas = LineAreas({"line1": 0.3}, source)
    function = estimate_hfs_temperature if source == "gaussian" else estimate_temperature
    with pytest.raises(ValueError, match="[Gg]aussian"):
        function(areas, feh=-0.2)
    with pytest.raises(TypeError):
        areas.areas["line2"] = 0.4


def test_gaussian_output_paths_reject_hfs_results(result, tmp_path):
    from sc_teff_method import plot_diagnostics

    for function in (summary_dict, plot_diagnostics):
        with pytest.raises(TypeError):
            function(result)
    with pytest.raises(TypeError):
        write_fit_outputs(result, tmp_path / "wrong-method")
    assert not (tmp_path / "wrong-method").exists()
    with pytest.raises(TypeError):
        hfs_summary_dict(object())


def test_gaussian_anchors_apply_to_actual_included_subset(result):
    anchors = {line: 0.08 for line in result.fits}
    anchors["line9"] = 0.01
    excluded = estimate_hfs_from_spectrum(
        result.spectrum, feh=-0.2, gaussian_anchor_areas=anchors, excluded_lines=["line9"],
    )
    assert excluded.strict_strength_domain_applicable is True
    assert excluded.temperature.n_lines == 8
    failing = estimate_hfs_from_spectrum(result.spectrum, feh=-0.2, gaussian_anchor_areas=anchors)
    assert failing.strict_strength_domain_applicable is False
    assert failing.temperature is not None
    assert any("false" in warning for warning in failing.warnings)
    all_available = estimate_hfs_from_spectrum(
        result.spectrum, feh=-0.2, validation_scope="all_available",
    )
    assert all_available.performance.scope == "all_available"
    assert all_available.strict_strength_domain_applicable is None


@pytest.mark.parametrize("options", [HFSFitOptions(max_nfev=1), None])
def test_failed_or_excluded_hfs_results_remain_missing(result, options):
    final = estimate_hfs_from_spectrum(
        result.spectrum, feh=-0.2, options=options,
        excluded_lines=tuple(result.fits) if options is None else (),
    )
    assert final.temperature is final.performance is None
    assert final.included_line_ids == ()
    assert not final.areas
    assert all(fit.area_angstrom is None for fit in final.fits.values())
    assert hfs_summary_dict(final)["temperature"] is None


def test_continuum_hfs_plot_uses_working_flux_and_preserves_source(result):
    original = result.spectrum
    rest = result.fits["line2"].rest_wavelength_air_angstrom
    local = np.abs(original.wavelength_air_angstrom-rest) <= 4
    flux = np.array(original.flux)
    flux[local] *= 1.035
    supplied = replace(original, flux=flux)
    corrected = estimate_hfs_from_spectrum(
        supplied, feh=-0.2, continuum_corrections={"line2": ContinuumCorrection(
            enabled=True, mode="constant", level=1.035,
        )},
    )
    figure = plot_hfs_diagnostics(corrected, spectrum_label="Invented local continuum")
    try:
        profile, residual = figure.axes[2:4]
        indices = np.searchsorted(original.wavelength_air_angstrom, profile.lines[1].get_xdata())
        np.testing.assert_allclose(profile.lines[1].get_ydata(), original.flux[indices], atol=2e-16)
        assert np.max(np.abs(residual.lines[0].get_ydata())) < 0.02
        assert len(profile.collections) == 1
    finally:
        figure.clear()
    np.testing.assert_array_equal(supplied.flux, flux)
    repeated = estimate_hfs_from_spectrum(
        supplied, feh=-0.2, continuum_corrections=corrected.continuum_corrections,
    )
    assert repeated.fits == corrected.fits


def test_hfs_json_csv_and_figures_preserve_method_and_missingness(result, tmp_path):
    final = estimate_hfs_from_spectrum(result.spectrum, feh=-0.2, excluded_lines=["line3"])
    files = write_hfs_outputs(final, tmp_path / "outputs", plot_format="both")
    summary = json.loads(files["summary.json"].read_text())
    assert summary["format"] == "sc-teff-hfs-fit"
    assert summary["model"]["weighting"] == "unweighted"
    assert len(summary["model"]["free_parameters"]) == 3
    assert sum(summary["atomic"]["component_counts"].values()) == 133
    assert summary["calibration"]["measurement_source"] == "hfs"
    assert summary["temperature"]["temperature_k"] == final.temperature.temperature_k
    assert summary["fits"]["line3"]["area_angstrom"] is None
    assert summary["strict_strength_domain_applicable"] is None
    assert "amplitude-width" in summary["formal_fit_uncertainty_note"]
    assert "wavelength_air_angstrom" not in summary["input"]
    with files["lines.csv"].open() as stream:
        rows = {row["line_id"]: row for row in csv.DictReader(stream)}
    assert len(rows) == 9
    assert rows["line3"]["area_angstrom"] == ""
    assert rows["line1"]["measurement_source"] == "hfs"
    assert float(rows["line1"]["area_angstrom"]) == final.fits["line1"].area_angstrom
    assert files["diagnostics.png"].read_bytes().startswith(b"\x89PNG")
    assert files["diagnostics.pdf"].read_bytes().startswith(b"%PDF")
    with pytest.raises(FileExistsError):
        write_hfs_outputs(final, tmp_path / "outputs")


@pytest.fixture
def input_file(result, tmp_path):
    path = tmp_path / "synthetic.csv"
    np.savetxt(path, np.column_stack((result.spectrum.wavelength_air_angstrom,
                                     result.spectrum.flux, result.spectrum.uncertainty,
                                     result.spectrum.mask)),
               delimiter=",", header="wave,flux,error,exclude", comments="", fmt="%.17g")
    return path


def command(source, output):
    return ["fit-hfs", str(source), *CLI_DECLARATIONS, "--feh", "-0.2", "--delimiter", ",",
            "--header", "--wavelength-column", "wave", "--flux-column", "flux",
            "--uncertainty-column", "error", "--mask-column", "exclude",
            "--output-dir", str(output)]


def test_hfs_cli_runs_and_protects_inputs_and_configs(input_file, tmp_path):
    output = tmp_path / "cli-hfs"
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({f"line{i}": 0.2 for i in range(1, 10)}))
    continuum = tmp_path / "continuum.json"
    continuum.write_text(json.dumps({"line1": {"mode": "constant", "enabled": True, "level": 1}}))
    run = subprocess.run([sys.executable, "-m", "sc_teff_method.cli",
                          *command(input_file, output), "--gaussian-anchor-areas", str(anchors),
                          "--continuum-config", str(continuum)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "HFS Teff" in run.stdout and "unweighted" in run.stdout
    summary = json.loads((output / "summary.json").read_text())
    assert summary["strict_strength_domain_applicable"] is True
    assert summary["input"]["sha256"]
    assert summary["continuum_corrections"]["line1"]["level"] == 1
    protected = tmp_path / "lines.csv"
    protected.write_bytes(input_file.read_bytes())
    assert main([*command(protected, tmp_path), "--overwrite"]) == 2
    assert protected.read_bytes() == input_file.read_bytes()
    protected_config = tmp_path / "summary.json"
    protected_config.write_text(json.dumps({"options": {}}))
    assert main([*command(input_file, tmp_path), "--fit-config", str(protected_config),
                 "--overwrite"]) == 2
    assert json.loads(protected_config.read_text()) == {"options": {}}


@pytest.mark.parametrize("extra", [["--feh", "nan"], ["--configuration", "unknown"]])
def test_invalid_hfs_cli_does_not_publish(input_file, tmp_path, extra):
    output = tmp_path / "invalid"
    try:
        code = main([*command(input_file, output), *extra])
    except SystemExit as exc:
        code = exc.code
    assert code == 2
    assert not output.exists()
