"""Entirely analytical toy spectra for examples and numerical verification.

Only the public Sc I line wavelengths are used. Amplitudes, widths, offsets,
sampling, noise and contaminants below are invented mathematical choices,
unrelated to individual observed spectra. These profiles do not model the
astrophysical complexity of a cool-giant atmosphere or demonstrate transfer
accuracy for an instrument.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

try:
    import numpy as np
except ImportError as exc:
    raise ImportError("Synthetic examples require sc-teff-method[spectra]") from exc

from ._data import get_calibration

SYNTHETIC_CASES = ("clean", "noise", "missing", "blends", "rv_offset", "variable")
DEFAULT_SYNTHETIC_SEED = 20261023
_SPEED_OF_LIGHT_KMS = 299792.458


@dataclass(frozen=True)
class SyntheticLine:
    """Invented Gaussian truth in the rest frame; area is in Angstrom."""

    amplitude: float
    centre_air_angstrom: float
    sigma_angstrom: float
    area_angstrom: float


@dataclass(frozen=True)
class SyntheticCase:
    """Raw input arrays and rest-frame analytic truth for one toy scenario.

    ``wavelength_air_angstrom`` is the raw, potentially redshifted input grid.
    Pass ``rv_kms`` explicitly to the spectrum loader or array preparation API
    to correct the ``rv_offset`` scenario. The other scenarios have no velocity
    shift. Missing spectral coverage does not remove the declared line truth.
    ``mask`` uses True to mean an excluded input pixel.
    """

    name: str
    wavelength_air_angstrom: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray | None
    mask: np.ndarray
    truth: Mapping[str, SyntheticLine]
    rv_kms: float | None
    description: str
    seed: int


def _read_only(values, *, dtype=float) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def make_synthetic_spectrum(
    case: str = "clean", *, seed: int = DEFAULT_SYNTHETIC_SEED,
) -> SyntheticCase:
    """Generate one of six deterministic, entirely artificial spectra.

    Each available line has its own +/-5 Angstrom segment. Large gaps between
    those segments are intentional; no interpolation or observation is used.
    The clean case uses ``A = 0.42 + 0.018*i``,
    ``sigma = 0.42 + 0.025*(i % 3)`` Angstrom and centre offsets
    ``0.06*((i % 3)-1)`` Angstrom for zero-based line index ``i``.

    ``noise`` adds independent Gaussian flux noise with known sigma 0.003.
    ``missing`` omits the line3 and line7 wavelength segments.
    ``blends`` adds two invented Gaussian neighbours and a deliberately wrong
    local continuum around line6. ``rv_offset`` redshifts the clean grid by
    +65 km/s using the relativistic Doppler factor. ``variable`` changes each
    line's width and sampling independently, including deliberately undersampled
    narrow lines. Every seed controls an independent NumPy random generator.
    """
    if not isinstance(case, str) or case not in SYNTHETIC_CASES:
        raise ValueError(f"case must be one of {SYNTHETIC_CASES}")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    wavelengths = {
        line_id: line.wavelength_air_angstrom
        for line_id, line in get_calibration().lines.items()
    }
    variable_steps = (0.025, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.30, 0.40)
    variable_widths = (0.14, 0.20, 0.26, 0.34, 0.42, 0.55, 0.65, 0.20, 0.12)
    truth = {}
    segments = []
    for i, (line_id, rest_wavelength) in enumerate(wavelengths.items()):
        amplitude = 0.42 + 0.018 * i
        sigma = variable_widths[i] if case == "variable" else 0.42 + 0.025 * (i % 3)
        centre = rest_wavelength + 0.06 * ((i % 3) - 1)
        truth[line_id] = SyntheticLine(
            amplitude=amplitude,
            centre_air_angstrom=centre,
            sigma_angstrom=sigma,
            area_angstrom=math.sqrt(2 * math.pi) * amplitude * sigma,
        )
        if case == "missing" and line_id in ("line3", "line7"):
            continue
        step = variable_steps[i] if case == "variable" else 0.04
        # Exact endpoints avoid a partially covered segment from arange rounding.
        intervals = round(10 / step)
        segments.append(rest_wavelength + np.linspace(-5, 5, intervals + 1))
    rest_grid = np.concatenate(segments)
    flux = np.ones_like(rest_grid)
    for line in truth.values():
        flux -= line.amplitude * np.exp(
            -0.5 * ((rest_grid - line.centre_air_angstrom) / line.sigma_angstrom)**2
        )
    uncertainty = None
    if case == "noise":
        uncertainty = np.full(rest_grid.shape, 0.003)
        flux += rng.normal(0, 0.003, rest_grid.size)
    if case == "blends":
        for line_id, amplitude, offset, width in (
            ("line4", 0.23, 0.75, 0.20), ("line8", 0.20, -0.85, 0.30),
        ):
            centre = wavelengths[line_id] + offset
            flux -= amplitude * np.exp(-0.5 * ((rest_grid - centre) / width)**2)
        local = np.abs(rest_grid - wavelengths["line6"]) <= 5
        flux[local] += 0.03 + 0.012 * (rest_grid[local] - wavelengths["line6"]) / 5
    rv_kms = 65.0 if case == "rv_offset" else None
    if rv_kms is None:
        raw_grid = rest_grid
    else:
        beta = rv_kms / _SPEED_OF_LIGHT_KMS
        raw_grid = rest_grid * math.sqrt((1 + beta) / (1 - beta))
    descriptions = {
        "clean": "Analytical Gaussian absorption lines on a unity continuum; no noise.",
        "noise": "Analytical Gaussian lines plus independent N(0, 0.003^2) flux noise.",
        "missing": "Analytical Gaussian lines with line3 and line7 wavelength coverage omitted.",
        "blends": (
            "Analytical lines with invented Gaussian blends at line4/line8 and an imposed "
            "local continuum error at line6."
        ),
        "rv_offset": "Analytical clean spectrum redshifted by +65 km/s; positive means recession.",
        "variable": (
            "Analytical lines with independently varied widths and sampling, including "
            "deliberately undersampled narrow lines."
        ),
    }
    return SyntheticCase(
        name=case,
        wavelength_air_angstrom=_read_only(raw_grid),
        flux=_read_only(flux),
        uncertainty=None if uncertainty is None else _read_only(uncertainty),
        mask=_read_only(np.zeros(rest_grid.shape, dtype=bool), dtype=bool),
        truth=MappingProxyType(truth),
        rv_kms=rv_kms,
        description=descriptions[case],
        seed=seed,
    )


def make_review_demo(*, seed: int = DEFAULT_SYNTHETIC_SEED) -> SyntheticCase:
    """Build an analytical inspection exercise without changing the six PR 2 cases.

    Start from the invented ``blends`` profiles, add independent Gaussian noise
    with sigma 0.0015, omit the lower part of line3's window, and explicitly mask
    a wing interval of line2. Line4 has an invented neighbouring absorption
    feature; line6 deliberately has the wrong continuum and should be rejected
    in the example. These choices are mathematical, not observation-derived.
    """
    original = make_synthetic_spectrum("blends", seed=seed)
    wavelength = original.wavelength_air_angstrom
    references = get_calibration().lines
    line3 = references["line3"].wavelength_air_angstrom
    keep = ~((wavelength >= line3-5) & (wavelength < line3-1.8))
    wavelength = wavelength[keep]
    noise_sigma = 0.0015
    rng = np.random.default_rng(original.seed)
    flux = original.flux[keep] + rng.normal(0, noise_sigma, wavelength.size)
    line2 = references["line2"].wavelength_air_angstrom
    mask = (wavelength >= line2+1.35) & (wavelength <= line2+1.65)
    return SyntheticCase(
        name="review",
        wavelength_air_angstrom=_read_only(wavelength),
        flux=_read_only(flux),
        uncertainty=_read_only(np.full(wavelength.shape, noise_sigma)),
        mask=_read_only(mask, dtype=bool),
        truth=original.truth,
        rv_kms=None,
        description=(
            "Analytical Gaussian review exercise: invented neighbours at line4/line8, "
            "a deliberately wrong continuum at line6, incomplete line3 coverage, a "
            "pre-masked line2 wing, and independent N(0, 0.0015^2) noise."
        ),
        seed=original.seed,
    )
