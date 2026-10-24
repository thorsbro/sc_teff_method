"""Independently evaluated analytical HFS mixtures for public examples/tests.

No observed spectrum, fit or continuum template is used. Only the public atomic
patterns are reused; every stellar profile parameter, blend and noise seed below
is invented. This is not atmosphere synthesis or instrument-transfer validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .hfs_data import load_hfs_atomic_data
from .spectrum import SPEED_OF_LIGHT_KMS

HFS_SYNTHETIC_CASES = ("clean", "noise", "missing", "blends", "rv_offset", "variable")
DEFAULT_HFS_SYNTHETIC_SEED = 20261025


@dataclass(frozen=True)
class HFSLineTruth:
    """Invented common-mixture parameters and infinite-profile analytic area."""

    amplitude: float
    centre_air_angstrom: float
    sigma_angstrom: float
    area_angstrom: float


@dataclass(frozen=True)
class SyntheticHFSCase:
    """Raw illustrative input arrays plus known rest-frame mathematical truth."""

    name: str
    wavelength_air_angstrom: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray | None
    mask: np.ndarray
    truth: Mapping[str, HFSLineTruth]
    rv_kms: float | None
    atomic_dataset_id: str
    description: str
    seed: int


def _frozen(array, *, dtype=float):
    values = np.array(array, dtype=dtype, copy=True)
    values.setflags(write=False)
    return values


def make_hfs_synthetic_spectrum(
    case: str = "clean", *, seed: int = DEFAULT_HFS_SYNTHETIC_SEED,
) -> SyntheticHFSCase:
    """Make deterministic invented Gaussian-component absorption spectra.

    The profile is independently evaluated as a loop over analytical component
    Gaussians, using the exact rational wavelength-offset equation directly.
    It does not call the fitting engine or its profile implementation. Each
    native segment spans +/-5 Å. The clean parameters are A=0.48+0.017*i,
    sigma=0.30+0.025*(i%3) Å and parent shifts 0.07*((i%3)-1) Å.

    ``noise`` supplies independent N(0,0.0025^2) noise and corresponding input
    sigma values. ``missing`` omits line3 and line7. ``blends`` invents Gaussian
    neighbours at line4/line8 plus masked continuum/wing pixels at line2.
    ``rv_offset`` redshifts the raw grid by +67 km/s; correct it explicitly.
    ``variable`` changes intrinsic widths and sampling independently and includes
    poorly sampled cases. None of these teach observational temperature accuracy.
    """
    if not isinstance(case, str) or case not in HFS_SYNTHETIC_CASES:
        raise ValueError(f"case must be one of {HFS_SYNTHETIC_CASES}")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    atomic = load_hfs_atomic_data()
    segments, truth = [], {}
    widths = (0.12, 0.18, 0.24, 0.31, 0.42, 0.55, 0.66, 0.12, 0.10)
    steps = (0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40)
    for i, (line, definition) in enumerate(atomic.lines.items()):
        rest = definition.wavelength_air_angstrom
        amplitude = 0.48+0.017*i
        sigma = widths[i] if case == "variable" else 0.30+0.025*(i % 3)
        centre = rest+0.07*((i % 3)-1)
        truth[line] = HFSLineTruth(amplitude, centre, sigma, math.sqrt(2*math.pi)*amplitude*sigma)
        if case == "missing" and line in ("line3", "line7"):
            continue
        step = steps[i] if case == "variable" else 0.025
        segments.append(rest+np.linspace(-5, 5, round(10/step)+1))
    x = np.concatenate(segments)
    flux = np.ones(x.shape)
    for line, definition in atomic.lines.items():
        parameters = truth[line]
        components = atomic.components[line]
        total = math.fsum(component.strength for component in components)
        for component in components:
            offset = (
                -definition.wavelength_air_angstrom*component.offset_cm1
                / (definition.cog_wavenumber_cm1+component.offset_cm1)
            )
            flux -= parameters.amplitude*(component.strength/total)*np.exp(
                -(x-parameters.centre_air_angstrom-offset)**2/(2*parameters.sigma_angstrom**2)
            )
    uncertainty = None
    if case == "noise":
        noise_sigma = 0.0025
        flux += np.random.default_rng(seed).normal(0, noise_sigma, x.size)
        uncertainty = np.full(x.shape, noise_sigma)
    mask = np.zeros(x.shape, dtype=bool)
    if case == "blends":
        for line, amplitude, offset, sigma in (
            ("line4", 0.22, 0.90, 0.17), ("line8", 0.16, -0.80, 0.20),
        ):
            centre = atomic.lines[line].wavelength_air_angstrom+offset
            flux -= amplitude*np.exp(-(x-centre)**2/(2*sigma**2))
        rest = atomic.lines["line2"].wavelength_air_angstrom
        mask = (x >= rest+1.3) & (x <= rest+1.6)
    rv = 67.0 if case == "rv_offset" else None
    if rv is not None:
        beta = rv/SPEED_OF_LIGHT_KMS
        x = x*math.sqrt((1+beta)/(1-beta))
    return SyntheticHFSCase(
        name=case, wavelength_air_angstrom=_frozen(x), flux=_frozen(flux),
        uncertainty=None if uncertainty is None else _frozen(uncertainty),
        mask=_frozen(mask, dtype=bool), truth=MappingProxyType(truth), rv_kms=rv,
        atomic_dataset_id=atomic.dataset_id,
        description=(
            f"Entirely analytical HFS example '{case}'; invented common amplitudes/widths, "
            "optional artificial blends or independently generated noise. No observational "
            "template or stellar fit was used."
        ),
        seed=seed,
    )
