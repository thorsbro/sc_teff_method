"""Explicitly masked synthetic NaNs stay missing in every output product."""
# ruff: noqa: E402

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from sc_teff_method import fit_spectrum, get_calibration, write_fit_outputs


def test_all_masked_nonfinite_arrays_produce_diagnostics_without_invented_values(tmp_path):
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    wavelength = rest + np.linspace(-5, 5, 201)
    result = fit_spectrum(
        wavelength, np.full(201, np.nan), uncertainty=np.full(201, np.nan),
        mask=np.ones(201, dtype=bool), wavelength_unit="angstrom", wavelength_frame="air",
        continuum_normalised=True, feh=-0.2,
    )
    assert result.temperature is None
    files = write_fit_outputs(result, tmp_path / "masked", plot_format="both",
                               spectrum_label="Analytical all-masked input")
    payload = files["summary.json"].read_text()
    summary = json.loads(payload)
    assert "NaN" not in payload
    assert summary["temperature"] is None
    assert summary["subset_performance"] is None
    assert summary["n_included_lines"] == 0
    assert summary["input"]["n_explicitly_masked"] == 201
    assert all(fit["area_angstrom"] is None for fit in summary["fits"].values())
    assert files["diagnostics.png"].stat().st_size > 0
    assert files["diagnostics.pdf"].stat().st_size > 0
