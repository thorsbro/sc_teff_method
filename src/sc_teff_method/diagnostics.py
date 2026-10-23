"""Local nine-line diagnostics. This module contains no spectral datasets."""

from __future__ import annotations

import numpy as np

from ._data import LINE_IDS
from .fitting import gaussian_absorption
from .workflow import SpectralEstimate


def plot_diagnostics(result: SpectralEstimate, *, spectrum_label: str = "Input spectrum"):
    """Return a headless Matplotlib Figure with profiles, masks and residuals.

    The caller owns the input spectrum and output figure. Repository example
    figures must be generated entirely from analytic synthetic spectra.
    Excluded nonfinite pixels are marked at continuum unity in profile panels.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    if not isinstance(result, SpectralEstimate):
        raise TypeError("result must be a SpectralEstimate")
    figure = Figure(figsize=(13.8, 12))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        3, 3, left=0.065, right=0.99, bottom=0.07, top=0.90,
        hspace=0.50, wspace=0.25,
    )
    spectrum = result.spectrum
    wavelength, flux = spectrum.wavelength_air_angstrom, spectrum.flux
    for index, line_id in enumerate(LINE_IDS):
        row, col = divmod(index, 3)
        pair = grid[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.20)
        profile = figure.add_subplot(pair[0])
        residual = figure.add_subplot(pair[1], sharex=profile)
        fit = result.fits[line_id]
        lower, upper = fit.window
        padding = 0.12 * (upper - lower)
        region = (wavelength >= lower - padding) & (wavelength <= upper + padding)
        in_window = (wavelength >= lower) & (wavelength <= upper)
        excluded = np.array(spectrum.mask, copy=True)
        for start, stop in fit.excluded_regions:
            excluded |= (wavelength >= start) & (wavelength <= stop)
        used = region & in_window & ~excluded & np.isfinite(flux)
        removed = region & excluded
        if fit.status == "excluded":
            used[:] = False
            removed |= region & in_window
        finite = region & np.isfinite(flux)
        profile.plot(wavelength[finite], flux[finite], ".", color="0.65", ms=2)
        profile.plot(wavelength[used], flux[used], ".", color="#176b9b", ms=3)
        profile.plot(
            wavelength[removed], np.where(np.isfinite(flux[removed]), flux[removed], 1.0),
            "x", color="#bf3c39", ms=4,
        )
        if fit.parameters is not None:
            x_model = np.linspace(lower, upper, 400)
            profile.plot(x_model, gaussian_absorption(x_model, fit.parameters),
                         color="#e38a17", lw=1.5)
            model = gaussian_absorption(wavelength[region], fit.parameters)
            delta = flux[region] - model
            residual.plot(wavelength[region][used[region]], delta[used[region]],
                          ".", color="#176b9b", ms=2.5)
            keep_excluded = removed[region] & np.isfinite(delta)
            residual.plot(wavelength[region][keep_excluded], delta[keep_excluded],
                          "x", color="#bf3c39", ms=3)
            profile.axvline(fit.parameters.centre_air_angstrom, color="#e38a17", lw=0.8)
        else:
            residual.text(0.5, 0.5, "No fitted model", transform=residual.transAxes,
                          ha="center", va="center", fontsize=8, color="0.4")
        for axis in (profile, residual):
            axis.axvline(fit.rest_wavelength_air_angstrom, color="0.3", ls="--", lw=0.8)
            axis.axvline(lower, color="#8662a1", ls=":", lw=1)
            axis.axvline(upper, color="#8662a1", ls=":", lw=1)
            axis.set_xlim(lower - padding, upper + padding)
            axis.tick_params(labelsize=7)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        residual.axhline(0, color="0.5", lw=0.6)
        residual.set_ylabel("F−model", fontsize=7)
        residual.set_xlabel("Rest air wavelength (Å)", fontsize=8)
        profile.set_ylabel("Normalised flux", fontsize=8)
        profile.tick_params(labelbottom=False)
        included = "included" if line_id in result.included_line_ids else "excluded"
        area = "missing" if fit.area_angstrom is None else f"{fit.area_angstrom:.5f} Å"
        profile.set_title(f"{line_id}  {fit.status} / {included}\nArea: {area}", fontsize=9)
        flags = ", ".join(fit.flags) if fit.flags else "No numerical quality flags"
        if not np.any(region):
            flags = "No wavelength coverage"
        profile.text(0.02, 0.03, flags.replace(", ", "\n"), transform=profile.transAxes,
                     fontsize=6.3, va="bottom", color="#8e2927" if fit.flags else "0.3")
    handles = [
        Line2D([], [], color="#176b9b", marker=".", ls="", label="Fit pixels"),
        Line2D([], [], color="#bf3c39", marker="x", ls="", label="Excluded pixels"),
        Line2D([], [], color="#e38a17", label="Gaussian / fitted centre"),
        Line2D([], [], color="0.3", ls="--", label="Rest wavelength"),
        Line2D([], [], color="#8662a1", ls=":", label="Fitting window"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.949),
                  ncol=5, fontsize=8, frameon=False)
    temperature = ("No combined temperature" if result.temperature is None else
                   f"Teff = {result.temperature.temperature_k:.2f} K, "
                   f"{result.temperature.n_lines} lines")
    figure.suptitle(f"Sc I Gaussian fits — {spectrum_label}\n{temperature}", y=0.992, fontsize=13)
    figure.text(
        0.5, 0.015,
        "Continuum fixed at unity. Inspect all fits; convergence is not scientific acceptance. "
        "Default windows are provisional.", ha="center", fontsize=8,
    )
    return figure
