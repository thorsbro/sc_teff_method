"""Precise JSON/CSV and local diagnostic files, with explicit overwrite control."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from pathlib import Path

from . import __version__
from ._data import get_calibration
from .diagnostics import plot_diagnostics
from .models import SubsetPerformance, TemperatureEstimate
from .workflow import SpectralEstimate


def _plain(value):
    if is_dataclass(value):
        result = {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, (TemperatureEstimate, SubsetPerformance)):
            result["n_lines"] = value.n_lines
        return result
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def summary_dict(result: SpectralEstimate, *, input_metadata: dict | None = None) -> dict:
    """Create a numerical summary without embedding input wavelength/flux arrays."""
    if not isinstance(result, SpectralEstimate):
        raise TypeError(
            "Gaussian outputs require a SpectralEstimate; use write_hfs_outputs for HFS"
        )
    calibration = get_calibration(result.configuration)
    spectrum = result.spectrum
    summary = {
        "schema_version": 1,
        "software": {
            "name": "sc-teff-method", "version": __version__,
            **{name: version(name) for name in ("numpy", "scipy", "matplotlib")},
        },
        "status": "temperature_available" if result.temperature is not None else "no_usable_lines",
        "input": {
            **(input_metadata or {}), "wavelength_unit": "angstrom", "wavelength_frame": "air",
            "continuum_normalised": True, "n_pixels": len(spectrum.flux),
            "n_explicitly_masked": int(spectrum.mask.sum()),
            "rv_kms": spectrum.rv_kms, "doppler_factor": spectrum.doppler_factor,
            "rv_convention": "positive=recession; rest=input/sqrt((1+v/c)/(1-v/c))",
            "rest_wavelength_min_angstrom": float(spectrum.wavelength_air_angstrom[0]),
            "rest_wavelength_max_angstrom": float(spectrum.wavelength_air_angstrom[-1]),
            "flux_uncertainty_supplied": spectrum.uncertainty is not None,
        },
        "calibration": {key: getattr(calibration, key) for key in (
            "calibration_id", "configuration", "model_id", "temperature_scale_id",
            "measurement_source", "fit_snapshot", "validation_snapshot",
            "declared_strength_min_angstrom", "declared_strength_max_angstrom",
        )},
        "feh": result.feh,
        "reference_temperature_k": result.reference_temperature_k,
        "quality_policy": result.quality_policy,
        "fit_options": _plain(result.fit_options),
        "initial_parameter_overrides": _plain(result.initial_parameters),
        "included_line_ids": list(result.included_line_ids),
        "n_included_lines": len(result.included_line_ids),
        "exclusion_reasons": dict(result.exclusion_reasons),
        "fits": {line: {**_plain(fit), "requires_scientific_inspection": fit.requires_inspection}
                 for line, fit in result.fits.items()},
        "diagnostic_line_temperatures_k": dict(result.line_temperatures_k),
        "calibration_warnings": _plain(result.calibration_warnings),
        "temperature": _plain(result.temperature),
        "subset_performance": _plain(result.performance),
        "validation_scope": result.validation_scope,
        "validation_cohort": "exact_subset",
        "target_passes_declared_strength_cut": result.target_passes_declared_strength_cut,
        "warnings": list(result.warnings),
        "formal_fit_uncertainty_note": (
            "Local least-squares covariance errors assume the fixed continuum and Gaussian model. "
            "They exclude continuum errors, blends, correlated pixels, model mismatch and "
            "calibration uncertainty; they are not total temperature uncertainties."
        ),
    }
    if result.continuum_corrections is not None:
        summary["schema_version"] = 2
        summary["continuum_note"] = (
            "Entirely manual local continuum division, starting from the original input. "
            "Formal Gaussian covariance excludes continuum-placement uncertainty. Frozen "
            "calibration RMSE does not validate these corrections or post-hoc selection."
        )
        for line, correction in result.continuum_corrections.items():
            summary["fits"][line]["continuum"] = _plain(correction)
    return summary


def _write_lines(result: SpectralEstimate, path: Path) -> None:
    rows = []
    for line_id, fit in result.fits.items():
        params, errors = fit.parameters, fit.parameter_errors
        rows.append({
            "line_id": line_id, "status": fit.status,
            "included_in_temperature": line_id in result.included_line_ids,
            "exclusion_reason": result.exclusion_reasons.get(line_id, ""),
            "rest_wavelength_air_angstrom": fit.rest_wavelength_air_angstrom,
            "area_angstrom": fit.area_angstrom, "area_error_angstrom": fit.area_error_angstrom,
            "line_temperature_k": result.line_temperatures_k.get(line_id),
            "amplitude": params.amplitude if params else None,
            "amplitude_error": errors.amplitude if errors else None,
            "centre_air_angstrom": params.centre_air_angstrom if params else None,
            "centre_error_angstrom": errors.centre_air_angstrom if errors else None,
            "sigma_angstrom": params.sigma_angstrom if params else None,
            "sigma_error_angstrom": errors.sigma_angstrom if errors else None,
            "centre_displacement_angstrom": fit.centre_displacement_angstrom,
            "residual_rms": fit.residual_rms, "reduced_chi2": fit.reduced_chi2,
            "n_pixels": fit.n_pixels, "n_excluded": fit.n_excluded, "nfev": fit.nfev,
            "window_min_angstrom": fit.window[0], "window_max_angstrom": fit.window[1],
            "window_provisional": fit.window_provisional,
            "requires_scientific_inspection": fit.requires_inspection,
            "flags": "|".join(fit.flags),
            "calibration_warnings": "|".join(w.code for w in result.calibration_warnings
                                              if w.line_id == line_id),
            "message": fit.message,
        })
        if result.continuum_corrections is not None:
            correction = result.continuum_corrections[line_id]
            rows[-1].update({
                "continuum_enabled": correction.enabled,
                "continuum_mode": correction.mode,
                "continuum_constant_level": correction.level,
                "continuum_anchors": json.dumps(correction.anchors, allow_nan=False),
            })
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_fit_outputs(
    result: SpectralEstimate,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    plot_format: str = "png",
    spectrum_label: str = "Input spectrum",
    input_metadata: dict | None = None,
) -> dict[str, Path]:
    """Write summary.json, lines.csv and diagnostics.png and/or .pdf.

    Existing output directories are refused unless overwrite=True is explicit.
    Outputs are prepared before replacing files; unrelated existing files are
    retained. Missing numerical values are JSON null and empty CSV cells.
    """
    if plot_format not in ("png", "pdf", "both"):
        raise ValueError("plot_format must be png, pdf or both")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    destination = Path(output_dir).expanduser()
    if destination.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {destination}; use overwrite=True")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    formats = ("png", "pdf") if plot_format == "both" else (plot_format,)
    names = ["summary.json", "lines.csv", *(f"diagnostics.{ext}" for ext in formats)]
    with tempfile.TemporaryDirectory(prefix=".sc-teff-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "products"
        staging.mkdir()
        staging.joinpath("summary.json").write_text(
            json.dumps(summary_dict(result, input_metadata=input_metadata), indent=2,
                       allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        _write_lines(result, staging / "lines.csv")
        figure = plot_diagnostics(result, spectrum_label=spectrum_label)
        try:
            for extension in formats:
                figure.savefig(staging / f"diagnostics.{extension}", dpi=150)
        finally:
            figure.clear()
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory appeared during writing: {destination}")
            for name in names:
                os.replace(staging / name, destination / name)
        else:
            staging.rename(destination)
    return {name: destination / name for name in names}
