"""Explicit input preparation for user-supplied continuum-normalised spectra.

Only air wavelengths in Angstrom are supported. Nothing in this module infers
units, converts vacuum wavelengths, fits a continuum or interpolates pixels.
Masks use True to exclude a pixel. Non-finite flux/uncertainty values are allowed
only at explicitly masked pixels; wavelengths must always be finite and ordered.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

try:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
except ImportError as exc:
    raise ImportError(
        "Spectrum input requires the spectral dependencies; "
        "install sc-teff-method[spectra]"
    ) from exc

SPEED_OF_LIGHT_KMS = 299792.458


def _real_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional array of real numbers") from exc
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    # Check Python sequence members before NumPy can coerce a mixed bool/float
    # sequence into an ordinary floating dtype.
    if isinstance(values, (list, tuple)) and any(
        isinstance(value, (bool, np.bool_)) for value in values
    ):
        raise TypeError(f"{name} must contain real numbers, not booleans")
    if raw.dtype.kind not in "iuf":
        raise TypeError(
            f"{name} must contain real numbers, not booleans, strings or complex values"
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(raw, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be represented as real floating-point numbers") from exc
    return result


def _mask_vector(mask: ArrayLike | None, size: int) -> NDArray[np.bool_]:
    if mask is None:
        return np.zeros(size, dtype=bool)
    result = np.asarray(mask)
    if result.ndim != 1 or result.size != size:
        raise ValueError("mask must be one-dimensional and have the same length as wavelength")
    if result.dtype.kind != "b":
        raise TypeError("mask must contain booleans; True means exclude the pixel")
    return np.array(result, dtype=bool, copy=True)


def _velocity(value: Real | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("rv_kms must be a finite real number in km/s")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("rv_kms must be a finite real number in km/s") from exc
    if not math.isfinite(result) or abs(result) >= SPEED_OF_LIGHT_KMS:
        raise ValueError("rv_kms must be finite with absolute value below the speed of light")
    return result


def _doppler_factor(rv_kms: float | None) -> float:
    beta = 0.0 if rv_kms is None else rv_kms / SPEED_OF_LIGHT_KMS
    return math.sqrt((1 + beta) / (1 - beta))


@dataclass(frozen=True)
class Spectrum:
    """Validated, read-only input arrays on the rest-frame air-Angstrom grid.

    Prefer :func:`prepare_spectrum` or :func:`load_spectrum` to declare input
    conventions and apply an optional RV correction. Constructing this canonical
    record directly does not apply an RV correction. ``doppler_factor`` records
    the divisor already applied to input wavelengths. Flux and flux uncertainty
    are dimensionless and are never resampled or continuum corrected.
    """

    wavelength_air_angstrom: NDArray[np.float64] = field(repr=False)
    flux: NDArray[np.float64] = field(repr=False)
    uncertainty: NDArray[np.float64] | None = field(default=None, repr=False)
    mask: NDArray[np.bool_] | None = field(default=None, repr=False)
    rv_kms: float | None = None
    doppler_factor: float = 1.0

    def __post_init__(self) -> None:
        wavelength = _real_vector(self.wavelength_air_angstrom, "wavelength")
        flux = _real_vector(self.flux, "flux")
        if wavelength.size < 2:
            raise ValueError("A spectrum must contain at least two wavelength samples")
        if flux.size != wavelength.size:
            raise ValueError("wavelength and flux must have the same length")
        if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("wavelength must contain only positive finite air wavelengths")
        if np.any(np.diff(wavelength) <= 0):
            raise ValueError("wavelength must be strictly increasing, without duplicate pixels")
        mask = _mask_vector(self.mask, wavelength.size)
        if not np.all(np.isfinite(flux[~mask])):
            raise ValueError("flux must be finite on every unmasked pixel")
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = _real_vector(self.uncertainty, "uncertainty")
            if uncertainty.size != wavelength.size:
                raise ValueError("uncertainty must have the same length as wavelength")
            usable = uncertainty[~mask]
            if not np.all(np.isfinite(usable)) or np.any(usable <= 0):
                raise ValueError("uncertainty must be positive and finite on every unmasked pixel")
        velocity = _velocity(self.rv_kms)
        if (isinstance(self.doppler_factor, (bool, np.bool_))
                or not isinstance(self.doppler_factor, Real)
                or not math.isfinite(self.doppler_factor)
                or self.doppler_factor <= 0
                or not math.isclose(
                    self.doppler_factor, _doppler_factor(velocity), rel_tol=1e-14, abs_tol=0,
                )):
            raise ValueError("doppler_factor must match the already-applied rv_kms correction")
        for name, values in (
            ("wavelength_air_angstrom", wavelength), ("flux", flux),
            ("uncertainty", uncertainty), ("mask", mask),
        ):
            if values is not None:
                values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "rv_kms", velocity)
        object.__setattr__(self, "doppler_factor", float(self.doppler_factor))

    @property
    def n_pixels(self) -> int:
        return self.wavelength_air_angstrom.size


def prepare_spectrum(
    wavelength: ArrayLike,
    flux: ArrayLike,
    *,
    wavelength_unit: str,
    wavelength_frame: str,
    continuum_normalised: bool,
    uncertainty: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    rv_kms: Real | None = None,
) -> Spectrum:
    """Validate arrays and optionally correct a declared radial velocity.

    Explicitly pass ``wavelength_unit="angstrom"``, ``wavelength_frame="air"``
    and ``continuum_normalised=True``. Unsupported conventions are rejected.

    Positive ``rv_kms`` means recession/redshift. The correction divides input
    wavelengths by ``sqrt((1 + v/c)/(1 - v/c))``, with c = 299792.458 km/s.
    Negative velocity means blueshift. None performs no correction; zero is an
    explicitly declared zero correction. Flux, uncertainty and masks are copied
    without interpolation, renormalisation or automatic pixel rejection.
    """
    if wavelength_unit != "angstrom":
        raise ValueError("Declare wavelength_unit='angstrom'; other units are not converted")
    if wavelength_frame != "air":
        raise ValueError("Declare wavelength_frame='air'; vacuum wavelengths are not converted")
    if continuum_normalised is not True:
        raise ValueError("Declare continuum_normalised=True after preparing the continuum yourself")
    velocity = _velocity(rv_kms)
    factor = _doppler_factor(velocity)
    input_wavelength = _real_vector(wavelength, "wavelength")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rest_wavelength = input_wavelength / factor
    return Spectrum(
        wavelength_air_angstrom=rest_wavelength,
        flux=flux,
        uncertainty=uncertainty,
        mask=mask,
        rv_kms=velocity,
        doppler_factor=factor,
    )


def _column_selector(value: int | str | None, label: str, header: bool) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise TypeError(f"{label} must be a zero-based integer or a column name")
    if isinstance(value, str):
        if not value or not value.strip():
            raise ValueError(f"{label} cannot be empty")
        if not header:
            raise ValueError(f"{label} by name requires header=True")
    elif value < 0:
        raise ValueError(f"{label} must be a non-negative zero-based index")
    return value


def load_spectrum(
    path: str | Path,
    *,
    wavelength_column: int | str = 0,
    flux_column: int | str = 1,
    uncertainty_column: int | str | None = None,
    mask_column: int | str | None = None,
    delimiter: str | None = None,
    header: bool = False,
    skiprows: int = 0,
    wavelength_unit: str,
    wavelength_frame: str,
    continuum_normalised: bool,
    rv_kms: Real | None = None,
) -> Spectrum:
    """Read explicit numeric columns from whitespace ASCII or delimited CSV.

    ``delimiter=None`` means whitespace; CSV requires its declared single-character
    delimiter. No delimiter, header, units or velocity are inferred. Column
    indices are zero based; names require ``header=True``. ``skiprows`` counts
    physical lines before parsing, and full-line ``#`` comments are ignored.
    Empty rows, inconsistent row widths and missing selected fields are errors.
    A mask column contains numeric 0 (use) or 1 (exclude). Non-finite selected
    flux/uncertainty values still require an explicit mask. FITS is unsupported.
    """
    if not isinstance(header, bool):
        raise TypeError("header must be a boolean")
    if isinstance(skiprows, bool) or not isinstance(skiprows, Integral):
        raise TypeError("skiprows must be a non-negative integer")
    if skiprows < 0:
        raise ValueError("skiprows must be a non-negative integer")
    if delimiter is not None and (
        not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in "\r\n#"
    ):
        raise ValueError(
            "delimiter must be None (whitespace) or one character other than #/newline"
        )
    selectors = {
        "wavelength": _column_selector(wavelength_column, "wavelength_column", header),
        "flux": _column_selector(flux_column, "flux_column", header),
        "uncertainty": _column_selector(uncertainty_column, "uncertainty_column", header),
        "mask": _column_selector(mask_column, "mask_column", header),
    }
    if selectors["wavelength"] is None or selectors["flux"] is None:
        raise ValueError("wavelength_column and flux_column are required")
    path = Path(path)
    if path.suffix.lower() in (".fit", ".fits", ".fts") or path.name.lower().endswith(
        (".fits.gz", ".fit.gz", ".fts.gz")
    ):
        raise ValueError("FITS input is not supported; supply an ASCII or CSV spectrum")
    columns = {key: [] for key, selector in selectors.items() if selector is not None}
    indices = None
    width = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number <= skiprows or line.lstrip().startswith("#"):
                continue
            if not line.strip():
                raise ValueError(f"Empty spectrum row at line {line_number}")
            if delimiter is None:
                fields = line.split()
            else:
                try:
                    fields = next(csv.reader([line], delimiter=delimiter, strict=True))
                except csv.Error as exc:
                    raise ValueError(f"Malformed CSV at line {line_number}") from exc
                fields = [value.strip() for value in fields]
            if width is None:
                width = len(fields)
                if header:
                    if any(not name for name in fields) or len(set(fields)) != width:
                        raise ValueError("Header names must be non-empty and distinct")
                indices = {}
                for key, selector in selectors.items():
                    if selector is None:
                        continue
                    if isinstance(selector, str):
                        if selector not in fields:
                            raise ValueError(f"Unknown {key} column name: {selector}")
                        index = fields.index(selector)
                    else:
                        index = int(selector)
                    if index >= width:
                        raise ValueError(f"{key} column index is outside the {width}-column input")
                    indices[key] = index
                if header:
                    continue
            if len(fields) != width:
                raise ValueError(
                    f"Expected {width} columns at line {line_number}, got {len(fields)}"
                )
            for key, index in indices.items():
                if not fields[index]:
                    raise ValueError(f"Missing {key} value at line {line_number}")
                try:
                    value = float(fields[index])
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric {key} value at line {line_number}") from exc
                if key == "mask":
                    if not math.isfinite(value) or value not in (0.0, 1.0):
                        raise ValueError(f"Mask must be 0 or 1 at line {line_number}")
                    value = bool(value)
                columns[key].append(value)
    if not columns["wavelength"]:
        raise ValueError("Spectrum file has no data rows")
    return prepare_spectrum(
        columns["wavelength"], columns["flux"],
        uncertainty=columns.get("uncertainty"), mask=columns.get("mask"),
        wavelength_unit=wavelength_unit, wavelength_frame=wavelength_frame,
        continuum_normalised=continuum_normalised, rv_kms=rv_kms,
    )
