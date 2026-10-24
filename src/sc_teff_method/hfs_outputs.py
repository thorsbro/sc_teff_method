"""Local HFS-specific diagnostics and precise, spectrum-free JSON/CSV exports."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import __version__
from ._data import LINE_IDS
from .continuum import evaluate_continuum
from .hfs import hfs_absorption
from .hfs_calibration import get_hfs_calibration
from .hfs_data import load_hfs_atomic_data
from .hfs_workflow import HFSSpectralEstimate
from .outputs import _plain
from .review_io import protect_output_inputs


def _result(value):
    if not isinstance(value, HFSSpectralEstimate) or value.measurement_source != "hfs":
        raise TypeError("HFS outputs require an HFSSpectralEstimate, not Gaussian results")
    return value


def hfs_summary_dict(result: HFSSpectralEstimate, *, input_metadata=None) -> dict:
    """Record HFS identities, controls, areas and statistics without spectral arrays."""
    result = _result(result)
    calibration = get_hfs_calibration(result.configuration)
    atomic = load_hfs_atomic_data()
    spectrum = result.spectrum
    return {
        "format": "sc-teff-hfs-fit", "schema_version": 1, "measurement_source": "hfs",
        "software": {"name": "sc-teff-method", "version": __version__,
                     **{key: version(key) for key in ("numpy", "scipy", "matplotlib")}},
        "status": "temperature_available" if result.temperature else "no_usable_lines",
        "input": {
            **(input_metadata or {}), "wavelength_unit": "angstrom", "wavelength_frame": "air",
            "continuum_normalised": True, "n_pixels": spectrum.n_pixels,
            "n_explicitly_masked": int(spectrum.mask.sum()), "rv_kms": spectrum.rv_kms,
            "doppler_factor": spectrum.doppler_factor,
            "rv_convention": "positive=recession; rest=input/sqrt((1+v/c)/(1-v/c))",
            "flux_uncertainty_supplied": spectrum.uncertainty is not None,
        },
        "atomic": {
            "dataset_id": atomic.dataset_id, "policy_id": atomic.policy.policy_id,
            "isotope": "45Sc", "nuclear_spin_I": atomic.policy.I,
            "component_counts": {line: len(atomic.components[line]) for line in LINE_IDS},
            "source_sha256": dict(atomic.source_hashes),
            "note": "Fixed A+B positions; frozen published rounded strengths renormalised "
                    "as in the reference. Table 4 positions are an A-only-like comparison.",
        },
        "model": {
            "profile": "1 - A * sum(s_k * exp(-(lambda - centre - offset_k)^2 / (2*sigma^2)))",
            "area": "sqrt(2*pi) * A * sigma; full profile, sum(s_k)=1",
            "free_parameters": ["amplitude", "parent_centre_air_angstrom", "sigma_angstrom"],
            "weighting": "unweighted", "continuum_in_model": 1.0,
        },
        "calibration": {key: getattr(calibration, key) for key in (
            "calibration_id", "configuration", "model_id", "temperature_scale_id",
            "measurement_source", "fit_snapshot", "validation_snapshot",
            "selection_anchor_source", "selection_note",
        )},
        "feh": result.feh, "reference_temperature_k": result.reference_temperature_k,
        "fit_options": asdict(result.fit_options), "quality_policy": result.quality_policy,
        "initial_parameter_overrides": _plain(result.initial_parameters),
        "continuum_corrections": _plain(result.continuum_corrections),
        "continuum_options": asdict(result.continuum_options),
        "fits": {line: {**_plain(fit), "requires_scientific_inspection": fit.requires_inspection}
                 for line, fit in result.fits.items()},
        "diagnostic_line_temperatures_k": dict(result.line_temperatures_k),
        "included_line_ids": list(result.included_line_ids),
        "n_included_lines": len(result.included_line_ids),
        "exclusion_reasons": dict(result.exclusion_reasons),
        "temperature": _plain(result.temperature), "subset_performance": _plain(result.performance),
        "validation_scope": result.validation_scope, "validation_cohort": "exact_subset",
        "gaussian_anchor_areas": _plain(result.gaussian_anchor_areas),
        "strict_strength_domain_applicable": result.strict_strength_domain_applicable,
        "calibration_warnings": _plain(result.calibration_warnings),
        "warnings": list(result.warnings),
        "formal_fit_uncertainty_note": (
            "Residual-scaled unweighted covariance includes amplitude-width covariance. "
            "It excludes continuum placement, correlated noise, blends, profile mismatch and "
            "calibration systematics. It is not a total temperature uncertainty."
        ),
    }


def plot_hfs_diagnostics(result: HFSSpectralEstimate, *, spectrum_label="Input spectrum"):
    """Plot all nine HFS mixtures, corrected fit pixels, residuals and component positions."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    result = _result(result)
    atomic = load_hfs_atomic_data()
    figure = Figure(figsize=(13.8, 12))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(3, 3, left=0.065, right=0.98, bottom=0.08, top=0.865,
                              hspace=0.50, wspace=0.25)
    x = result.spectrum.wavelength_air_angstrom
    for index, line in enumerate(LINE_IDS):
        fit = result.fits[line]
        working = result.corrected_spectrum(line)
        pair = grid[index//3, index % 3].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.2)
        profile = figure.add_subplot(pair[0])
        residual = figure.add_subplot(pair[1], sharex=profile)
        lower, upper = fit.window
        padding = 0.12*(upper-lower)
        region = (x >= lower-padding) & (x <= upper+padding)
        inside = region & (x >= lower) & (x <= upper)
        excluded = np.array(working.mask, copy=True)
        for lo, hi in fit.excluded_regions:
            excluded |= (x >= lo) & (x <= hi)
        used = inside & ~excluded & np.isfinite(working.flux)
        removed = inside & excluded
        if fit.status == "excluded":
            used[:] = False
            removed |= inside
        native = region & np.isfinite(result.spectrum.flux)
        profile.plot(x[native], result.spectrum.flux[native], ".", color="0.65", ms=2)
        profile.plot(x[used], working.flux[used], ".", color="#176b9b", ms=3)
        profile.plot(x[removed], np.where(np.isfinite(working.flux[removed]),
                                         working.flux[removed], 1), "x", color="#bf3c39", ms=4)
        correction = result.continuum_corrections.get(line)
        if correction is not None and correction.active:
            cx = np.linspace(lower, upper, 100)
            profile.plot(cx, evaluate_continuum(cx, correction), "-.", color="#13826a", lw=1)
        if fit.parameters is not None:
            params, reference = fit.parameters, atomic.lines[line]

            def model(values):
                return hfs_absorption(
                    values, params.amplitude, params.centre_air_angstrom, params.sigma_angstrom,
                    atomic.components[line], parent_wavelength=reference.wavelength_air_angstrom,
                    cog_wavenumber_cm1=reference.cog_wavenumber_cm1,
                )

            mx = np.linspace(lower, upper, 500)
            profile.plot(mx, model(mx), color="#e38a17", lw=1.5)
            residual.plot(x[used], working.flux[used]-model(x[used]),
                          ".", color="#176b9b", ms=2.5)
            valid_removed = removed & np.isfinite(working.flux)
            residual.plot(x[valid_removed], working.flux[valid_removed]-model(x[valid_removed]),
                          "x", color="#bf3c39", ms=3)
            positions = params.centre_air_angstrom + np.asarray(fit.component_offsets_angstrom)
            profile.vlines(positions, 0.91, 1.0, transform=profile.get_xaxis_transform(),
                           color="#13826a", alpha=0.65, lw=0.7)
            profile.axvline(params.centre_air_angstrom, color="#e38a17", lw=0.8)
        else:
            residual.text(0.5, 0.5, "No fitted HFS model", transform=residual.transAxes,
                          ha="center", fontsize=8)
        for axis in (profile, residual):
            axis.axvline(fit.rest_wavelength_air_angstrom, color="0.3", ls="--", lw=0.8)
            axis.axvline(lower, color="#8662a1", ls=":", lw=1)
            axis.axvline(upper, color="#8662a1", ls=":", lw=1)
            axis.set_xlim(lower-padding, upper+padding)
            axis.tick_params(labelsize=7)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        residual.axhline(0, color="0.5", lw=0.6)
        residual.set_xlabel("Rest air wavelength (Å)", fontsize=8)
        residual.set_ylabel("Fc−HFS", fontsize=7)
        profile.set_ylabel("Normalised flux", fontsize=8)
        profile.tick_params(labelbottom=False)
        area = "missing" if fit.area_angstrom is None else f"{fit.area_angstrom:.5f} Å"
        state = "included" if line in result.included_line_ids else "excluded"
        profile.set_title(f"{line}: HFS / {fit.status} / {state}\nArea {area}", fontsize=8.5)
        flags = "\n".join(fit.flags) if fit.flags else "Inspect before scientific use"
        profile.text(0.02, 0.03, flags,
                     transform=profile.transAxes, fontsize=6.3, va="bottom",
                     color="#8e2927" if fit.flags else "0.3")
    handles = [
        Line2D([], [], color="#176b9b", marker=".", ls="",
               label="Fit pixels (corrected if chosen)"),
        Line2D([], [], color="0.65", marker=".", ls="", label="Original input"),
        Line2D([], [], color="#e38a17", label="Total HFS mixture / parent centre"),
        Line2D([], [], color="#13826a", marker="|", ls="", label="Fixed component centres"),
        Line2D([], [], color="#bf3c39", marker="x", ls="", label="Excluded pixels"),
        Line2D([], [], color="0.3", ls="--", label="Laboratory parent"),
        Line2D([], [], color="#8662a1", ls=":", label="Fit window"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.949),
                  ncol=4, fontsize=8, frameon=False)
    temperature = ("No combined HFS temperature" if result.temperature is None else
                   f"HFS Teff = {result.temperature.temperature_k:.2f} K, "
                   f"{result.temperature.n_lines} lines")
    figure.suptitle(f"Sc I fixed A+B HFS fits — {spectrum_label}\n{temperature}",
                   y=0.992, fontsize=13)
    figure.text(0.5, 0.02,
                "Unweighted HFS mixture, three free parameters; convergence is not acceptance.\n"
                "Strict-domain validation uses Gaussian anchors, not HFS areas. "
                "Formal errors exclude continuum/profile systematics.", ha="center", fontsize=8)
    return figure


def write_hfs_outputs(
    result: HFSSpectralEstimate, output_dir, *, overwrite=False, plot_format="png",
    spectrum_label="Input spectrum", input_metadata=None, protected_paths=(),
) -> dict[str, Path]:
    """Write HFS JSON/CSV/PNG or PDF with existing no-clobber/input protections."""
    result = _result(result)
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    if plot_format not in ("png", "pdf", "both"):
        raise ValueError("plot_format must be png, pdf or both")
    destination = Path(output_dir).expanduser()
    protect_output_inputs(destination, protected_paths, include_review=False)
    if destination.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {destination}; use overwrite=True")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    formats = ("png", "pdf") if plot_format == "both" else (plot_format,)
    names = ["summary.json", "lines.csv", *(f"diagnostics.{item}" for item in formats)]
    with tempfile.TemporaryDirectory(prefix=".sc-teff-hfs-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "products"
        staging.mkdir()
        summary = hfs_summary_dict(result, input_metadata=input_metadata)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False)+"\n",
            encoding="utf-8",
        )
        rows = []
        for line, fit in result.fits.items():
            params, errors = fit.parameters, fit.parameter_errors
            rows.append({
                "line_id": line, "measurement_source": "hfs", "status": fit.status,
                "included_in_temperature": line in result.included_line_ids,
                "exclusion_reason": result.exclusion_reasons.get(line, ""),
                "atomic_dataset_id": fit.atomic_dataset_id, "n_components": fit.n_components,
                "area_angstrom": fit.area_angstrom, "area_error_angstrom": fit.area_error_angstrom,
                "line_temperature_k": result.line_temperatures_k.get(line),
                "amplitude": params.amplitude if params else None,
                "parent_centre_air_angstrom": params.centre_air_angstrom if params else None,
                "sigma_angstrom": params.sigma_angstrom if params else None,
                "amplitude_error": errors.amplitude if errors else None,
                "centre_error_angstrom": errors.centre_air_angstrom if errors else None,
                "sigma_error_angstrom": errors.sigma_angstrom if errors else None,
                "residual_rms": fit.residual_rms, "diagnostic_reduced_chi2": fit.reduced_chi2,
                "n_pixels": fit.n_pixels, "n_excluded": fit.n_excluded,
                "window_lower": fit.window[0], "window_upper": fit.window[1],
                "weighting": fit.weighting, "flags": "|".join(fit.flags),
                "calibration_warnings": "|".join(
                    warning.code for warning in result.calibration_warnings
                    if warning.line_id == line
                ),
                "continuum": json.dumps(summary["continuum_corrections"].get(line),
                                        allow_nan=False, separators=(",", ":")),
                "gaussian_anchor_area_angstrom": (result.gaussian_anchor_areas or {}).get(line),
                "message": fit.message,
            })
        with (staging / "lines.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        figure = plot_hfs_diagnostics(result, spectrum_label=spectrum_label)
        try:
            for extension in formats:
                figure.savefig(staging / f"diagnostics.{extension}", dpi=150)
        finally:
            figure.clear()
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory appeared during writing: {destination}")
            for name in names:
                os.replace(staging/name, destination/name)
        else:
            staging.rename(destination)
    return {name: destination/name for name in names}
