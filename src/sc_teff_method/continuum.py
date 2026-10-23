"""Entirely manual local multiplicative continuum controls.

No continuum is inferred from data. Coordinates are the already corrected rest
frame in air Angstrom. Corrections are applied once to a working spectrum, never
to the caller's immutable original. Thresholds are provisional inspection prompts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .fitting import _interval, _regions
from .spectrum import Spectrum


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and strictly positive" if positive else ""))
    return value


@dataclass(frozen=True)
class ContinuumCorrection:
    """Manual none/constant/linear control; disabled controls retain their values.

    Constant mode uses ``level``. Linear mode uses exactly two manually supplied
    ``(wavelength_air_angstrom, continuum_level)`` anchors, in increasing order.
    Interpolation inside the anchors and linear extrapolation outside them use
    the same straight line. Extrapolation is flagged, never inferred from flux.
    """

    enabled: bool = False
    mode: str = "none"
    level: float = 1.0
    anchors: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("continuum enabled must be boolean")
        if self.mode not in ("none", "constant", "linear"):
            raise ValueError("continuum mode must be none, constant or linear")
        object.__setattr__(self, "level", _finite(self.level, "continuum level", positive=True))
        try:
            anchors = tuple(tuple(point) for point in self.anchors)
        except TypeError as exc:
            raise ValueError("continuum anchors must contain wavelength/level pairs") from exc
        if any(len(point) != 2 for point in anchors):
            raise ValueError("continuum anchors must contain wavelength/level pairs")
        anchors = tuple(
            (_finite(x, "anchor wavelength", positive=True),
             _finite(y, "anchor continuum", positive=True))
            for x, y in anchors
        )
        if self.mode == "linear" and len(anchors) != 2:
            raise ValueError("linear continuum requires exactly two anchors")
        if len(anchors) > 2 or (len(anchors) == 2 and anchors[0][0] >= anchors[1][0]):
            raise ValueError("continuum anchor wavelengths must strictly increase")
        object.__setattr__(self, "anchors", anchors)

    @property
    def active(self) -> bool:
        return self.enabled and self.mode != "none"


@dataclass(frozen=True)
class ContinuumOptions:
    """Configurable provisional continuum warning thresholds, not quality proof."""

    max_deviation_warning: float = 0.1
    warn_extrapolation: bool = True

    def __post_init__(self):
        object.__setattr__(self, "max_deviation_warning", _finite(
            self.max_deviation_warning, "max_deviation_warning", positive=True,
        ))
        if not isinstance(self.warn_extrapolation, bool):
            raise ValueError("warn_extrapolation must be boolean")


def evaluate_continuum(wavelength, correction: ContinuumCorrection):
    """Evaluate the chosen control without changing wavelengths or any flux.

    Linear extrapolation may produce non-positive values outside a fit region;
    :func:`apply_continuum` validates the actual fitting window before division.
    """
    if not isinstance(correction, ContinuumCorrection):
        raise TypeError("correction must be a ContinuumCorrection")
    x = np.asarray(wavelength)
    if x.dtype.kind not in "iuf" or not np.all(np.isfinite(x)):
        raise ValueError("continuum wavelengths must be finite real values")
    x = np.asarray(x, dtype=float)
    if not correction.active:
        return np.ones_like(x)
    if correction.mode == "constant":
        return np.full_like(x, correction.level)
    (x0, y0), (x1, y1) = correction.anchors
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return y0 + (x-x0) * ((y1-y0)/(x1-x0))


def apply_continuum(
    spectrum: Spectrum,
    correction: ContinuumCorrection,
    *,
    window,
    excluded_regions=(),
    options: ContinuumOptions | None = None,
):
    """Return a local working spectrum and continuum flags.

    Both flux and its supplied uncertainty are divided by C on window pixels.
    Positivity is checked even on masked pixels, avoiding unsafe display values.
    Pixels outside the window are unchanged. Wavelengths, masks and existing RV
    metadata are preserved; no preparation or Doppler correction is repeated.
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a Spectrum")
    if not isinstance(correction, ContinuumCorrection):
        raise TypeError("correction must be a ContinuumCorrection")
    if options is None:
        options = ContinuumOptions()
    if not isinstance(options, ContinuumOptions):
        raise TypeError("options must be ContinuumOptions")
    lower, upper = _interval(window, "continuum window", positive=True)
    _regions(excluded_regions, "excluded_regions")
    if not correction.active:
        return spectrum, ()
    x = spectrum.wavelength_air_angstrom
    region = (x >= lower) & (x <= upper)
    # Validate the continuous linear model at boundaries too, even if a window
    # has no data or a gap masks an invalid portion of the continuum.
    values = evaluate_continuum(np.r_[lower, x[region], upper], correction)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("continuum must be finite and strictly positive across the fitting window")
    flags = []
    if np.any(np.abs(values-1) > options.max_deviation_warning):
        flags.append("continuum_large_correction")
    if correction.mode == "linear" and options.warn_extrapolation:
        if lower < correction.anchors[0][0] or upper > correction.anchors[1][0]:
            flags.append("continuum_anchor_extrapolation")
    divisor = values[1:-1]
    flux = np.array(spectrum.flux, copy=True)
    uncertainty = (
        None if spectrum.uncertainty is None else np.array(spectrum.uncertainty, copy=True)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        flux[region] /= divisor
        if uncertainty is not None:
            uncertainty[region] /= divisor
    return Spectrum(
        wavelength_air_angstrom=x, flux=flux, uncertainty=uncertainty, mask=spectrum.mask,
        rv_kms=spectrum.rv_kms, doppler_factor=spectrum.doppler_factor,
    ), tuple(flags)
