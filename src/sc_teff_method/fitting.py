"""Fixed-continuum Gaussian measurements, with explicit numerical diagnostics.

Default windows and diagnostic thresholds are provisional, generic choices.
They have not been tuned on observed spectra. Convergence is never scientific
acceptance, and formal covariance excludes continuum and profile systematics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import TYPE_CHECKING

try:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
    from scipy.optimize import least_squares
except ImportError as exc:
    raise ImportError("Spectrum fitting requires 'sc-teff-method[spectra]'") from exc

from ._data import LINE_IDS, get_calibration

if TYPE_CHECKING:
    from .spectrum import Spectrum


def _real(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _interval(value, name: str, *, positive: bool = False) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain two finite endpoints")
    try:
        entries = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two finite endpoints") from exc
    if len(entries) != 2:
        raise ValueError(f"{name} must contain exactly two endpoints")
    lower, upper = (_real(v, name, positive=positive) for v in entries)
    if not lower < upper:
        raise ValueError(f"{name} endpoints must increase")
    return lower, upper


@dataclass(frozen=True)
class GaussianParameters:
    """Gaussian depth, centre (air Angstrom) and standard-deviation width.

    Also used for the three formal parameter errors, where centre and width
    entries are errors in Angstrom. Bounds/positivity are checked for initial
    fitting parameters rather than for error containers.
    """

    amplitude: float
    centre_air_angstrom: float
    sigma_angstrom: float

    def __post_init__(self) -> None:
        for name in ("amplitude", "centre_air_angstrom", "sigma_angstrom"):
            object.__setattr__(self, name, _real(getattr(self, name), name))


@dataclass(frozen=True)
class FitOptions:
    """Generic provisional fitting bounds and diagnostic thresholds.

    Windows default to +/-3 Angstrom and centres to +/-1 Angstrom about the
    published rest wavelength. Width bounds are 0.03--1.5 Angstrom in sigma.
    These are mathematical defaults, not observationally approved windows.
    Quality thresholds are inspection prompts, not acceptance criteria.
    """

    window_half_width_angstrom: float = 3.0
    centre_max_offset_angstrom: float = 1.0
    amplitude_bounds: tuple[float, float] = (1e-6, 1.0)
    sigma_bounds_angstrom: tuple[float, float] = (0.03, 1.5)
    min_pixels: int = 7
    max_nfev: int = 2000
    rms_warning: float = 0.02
    reduced_chi2_warning: float = 3.0
    centre_warning_angstrom: float = 0.5
    min_samples_per_fwhm: float = 4.0
    gap_factor: float = 3.0
    boundary_fraction: float = 0.01
    max_jacobian_condition: float = 1e10
    min_amplitude_snr: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "window_half_width_angstrom", "centre_max_offset_angstrom", "rms_warning",
            "reduced_chi2_warning", "centre_warning_angstrom", "min_samples_per_fwhm",
            "gap_factor", "boundary_fraction", "max_jacobian_condition", "min_amplitude_snr",
        ):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        for name in ("min_pixels", "max_nfev"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < (3 if name == "min_pixels" else 1):
                raise ValueError(f"{name} is too small")
            object.__setattr__(self, name, int(value))
        amplitude = _interval(self.amplitude_bounds, "amplitude_bounds")
        if amplitude[0] < 0:
            raise ValueError("amplitude_bounds must be non-negative")
        object.__setattr__(self, "amplitude_bounds", amplitude)
        object.__setattr__(
            self, "sigma_bounds_angstrom",
            _interval(self.sigma_bounds_angstrom, "sigma_bounds_angstrom", positive=True),
        )
        if not self.boundary_fraction < 0.5:
            raise ValueError("boundary_fraction must be less than 0.5")
        if self.gap_factor <= 1 or self.max_jacobian_condition <= 1:
            raise ValueError("gap_factor and max_jacobian_condition must exceed one")


@dataclass(frozen=True)
class LineFit:
    """One measurement or an explicit failed/unavailable/excluded line.

    Failed and unavailable fits have no fitted parameters or area. Quality flags
    accompany numerical convergence and must be considered by the caller's
    inclusion policy. Every converged fit still requires scientific inspection.
    """

    line_id: str
    rest_wavelength_air_angstrom: float
    window: tuple[float, float]
    window_provisional: bool
    status: str
    message: str
    parameters: GaussianParameters | None = None
    area_angstrom: float | None = None
    area_error_angstrom: float | None = None
    parameter_errors: GaussianParameters | None = None
    covariance: tuple[tuple[float, ...], ...] | None = None
    residual_rms: float | None = None
    n_pixels: int = 0
    n_excluded: int = 0
    centre_displacement_angstrom: float | None = None
    flags: tuple[str, ...] = ()
    nfev: int = 0
    reduced_chi2: float | None = None
    excluded_regions: tuple[tuple[float, float], ...] = ()
    weighting: str = "unweighted"

    @property
    def requires_inspection(self) -> bool:
        return self.status == "converged"


def gaussian_absorption(
    wavelength: ArrayLike, parameters: GaussianParameters,
) -> NDArray[np.float64]:
    """Evaluate ``1 - A exp[-(lambda-centre)^2 / (2 sigma^2)]``.

    This is an analytical profile, not a stellar spectral-synthesis model.
    Wavelengths and both positional parameters must use the same Angstrom frame.
    """
    if not isinstance(parameters, GaussianParameters):
        raise TypeError("parameters must be GaussianParameters")
    if parameters.amplitude < 0 or parameters.sigma_angstrom <= 0:
        raise ValueError("amplitude must be non-negative and sigma must be positive")
    x = np.asarray(wavelength)
    if x.dtype.kind not in "iuf" or not np.all(np.isfinite(x)):
        raise ValueError("wavelength must contain finite real values")
    x = x.astype(float, copy=False)
    return 1.0 - parameters.amplitude * np.exp(
        -0.5 * ((x - parameters.centre_air_angstrom) / parameters.sigma_angstrom)**2
    )


def gaussian_area(amplitude: Real, sigma: Real) -> float:
    """Positive analytic absorption area in Angstrom, integrated over all lambda."""
    amplitude = _real(amplitude, "amplitude")
    sigma = _real(sigma, "sigma", positive=True)
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")
    result = math.sqrt(2 * math.pi) * amplitude * sigma
    if not math.isfinite(result):
        raise ValueError("Gaussian area must be finite")
    return result


def _regions(value, name: str) -> tuple[tuple[float, float], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of intervals")
    try:
        return tuple(_interval(v, name, positive=True) for v in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of intervals") from exc


def fit_pixel_masks(
    spectrum: Spectrum,
    window: tuple[float, float],
    excluded_regions: Iterable[tuple[float, float]] = (),
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return full-array region and actually used masks (True selects a pixel).

    Window and exclusion endpoints are inclusive, in the RV-corrected rest-air
    Angstrom coordinates. The spectrum mask uses the opposite convention:
    True means excluded. The resulting used mask includes both exclusions.
    """
    lower, upper = _interval(window, "window", positive=True)
    excluded = _regions(excluded_regions, "excluded_regions")
    x = spectrum.wavelength_air_angstrom
    region = (x >= lower) & (x <= upper)
    used = region & ~spectrum.mask
    for lo, hi in excluded:
        used &= ~((x >= lo) & (x <= hi))
    region.setflags(write=False)
    used.setflags(write=False)
    return region, used


def _line_mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by line ID")
    if any(not isinstance(line, str) or line not in LINE_IDS for line in value):
        raise ValueError(f"Unknown line ID in {name}; use line1 through line9")
    return dict(value)


def _bounds(rest: float, window: tuple[float, float], options: FitOptions):
    centre_lo = max(window[0], rest - options.centre_max_offset_angstrom)
    centre_hi = min(window[1], rest + options.centre_max_offset_angstrom)
    if not centre_lo < centre_hi:
        raise ValueError("Fitting window does not overlap the permitted line-centre interval")
    reference = (window[0] + window[1]) / 2
    lower = np.array([options.amplitude_bounds[0], centre_lo-reference,
                      options.sigma_bounds_angstrom[0]])
    upper = np.array([options.amplitude_bounds[1], centre_hi-reference,
                      options.sigma_bounds_angstrom[1]])
    return reference, lower, upper


def _start(parameters: GaussianParameters, reference: float, lower, upper):
    if not isinstance(parameters, GaussianParameters):
        raise TypeError("Each initial_parameters value must be GaussianParameters")
    initial = np.array([
        parameters.amplitude, parameters.centre_air_angstrom-reference, parameters.sigma_angstrom,
    ])
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Initial Gaussian parameters are outside the fitting bounds")
    return initial


def _coverage_flags(raw_x, used_x, window, options: FitOptions) -> list[str]:
    flags = []
    spacing = float(np.median(np.diff(raw_x))) if len(raw_x) > 1 else window[1]-window[0]
    tolerance = 1.5 * spacing
    if raw_x[0] - window[0] > tolerance or window[1] - raw_x[-1] > tolerance:
        flags.append("partial_window_coverage")
    if len(raw_x) > 2:
        if np.any(np.diff(raw_x) > 1.5 * spacing):
            flags.append("missing_or_irregular_pixels")
        if np.any(np.diff(raw_x) > options.gap_factor * spacing):
            flags.append("internal_gap")
    if len(used_x):
        if used_x[0] - raw_x[0] > tolerance or raw_x[-1] - used_x[-1] > tolerance:
            flags.append("masked_window_edge")
        if len(used_x) > 1 and len(used_x) < len(raw_x):
            used_steps = np.diff(used_x)
            if np.any(used_steps > 1.5 * spacing):
                flags.append("masked_gap")
    return flags


def _covariance(jacobian, residuals, parameters, *, absolute_sigma: bool,
                at_boundary: bool, options: FitOptions):
    if at_boundary:
        return None, None, None, ["covariance_unavailable_at_boundary"]
    n = len(residuals)
    if n <= 3:
        return None, None, None, ["covariance_insufficient_dof"]
    try:
        _, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None, ["covariance_singular"]
    threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0]
    if not np.all(np.isfinite(singular)) or singular[-1] <= threshold:
        return None, None, None, ["covariance_singular"]
    if singular[0] / singular[-1] > options.max_jacobian_condition:
        return None, None, None, ["covariance_ill_conditioned"]
    covariance = (vh.T / singular**2) @ vh
    if not absolute_sigma:
        covariance *= float(np.dot(residuals, residuals)) / (n-3)
    gradient = math.sqrt(2 * math.pi) * np.array([
        parameters.sigma_angstrom, 0.0, parameters.amplitude,
    ])
    variance = float(gradient @ covariance @ gradient)
    if (not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < 0)
            or not math.isfinite(variance) or variance < 0):
        return None, None, None, ["covariance_invalid"]
    errors = GaussianParameters(*np.sqrt(np.diag(covariance)))
    packed = tuple(tuple(float(value) for value in row) for row in covariance)
    return packed, errors, math.sqrt(variance), []


def _fit_one(spectrum, line_id, rest, window, provisional, excluded, initial, options):
    region, used = fit_pixel_masks(spectrum, window, excluded)
    raw_x = spectrum.wavelength_air_angstrom[region]
    x = spectrum.wavelength_air_angstrom[used]
    y = spectrum.flux[used]
    n = len(x)
    common = dict(
        line_id=line_id, rest_wavelength_air_angstrom=rest, window=window,
        window_provisional=provisional, n_pixels=n, n_excluded=int(np.sum(region & ~used)),
        excluded_regions=excluded,
        weighting="unweighted" if spectrum.uncertainty is None else "inverse_sigma",
    )
    if not len(raw_x):
        return LineFit(status="unavailable", message="No wavelength coverage in fitting window",
                       flags=("no_coverage",), **common)
    flags = _coverage_flags(raw_x, x, window, options)
    if n < options.min_pixels:
        return LineFit(status="unavailable", message="Too few unmasked native pixels",
                       flags=tuple(flags + ["insufficient_samples"]), **common)
    reference, lower, upper = _bounds(rest, window, options)
    shifted = x - reference
    if initial is None:
        eligible = (shifted >= lower[1]) & (shifted <= upper[1])
        centre = float(x[eligible][np.argmin(y[eligible])]) if np.any(eligible) else rest
        start = np.array([max(1.0 - float(np.min(y)), 0.001), centre-reference, 0.35])
        epsilon = np.maximum((upper - lower) * 1e-8, np.finfo(float).eps)
        start = np.clip(start, lower+epsilon, upper-epsilon)
    else:
        start = _start(initial, reference, lower, upper)
    weights = 1.0 if spectrum.uncertainty is None else 1.0 / spectrum.uncertainty[used]

    def model(parameters):
        amplitude, mu, sigma = parameters
        return 1.0 - amplitude * np.exp(-0.5 * ((shifted-mu) / sigma)**2)

    def residual(parameters):
        return (model(parameters)-y) * weights

    def jacobian(parameters):
        amplitude, mu, sigma = parameters
        distance = shifted-mu
        exponential = np.exp(-0.5 * (distance/sigma)**2)
        matrix = np.column_stack((
            -exponential,
            -amplitude * exponential * distance / sigma**2,
            -amplitude * exponential * distance**2 / sigma**3,
        ))
        return matrix * weights if np.isscalar(weights) else matrix * weights[:, None]

    try:
        fitted = least_squares(
            residual, start, jac=jacobian, bounds=(lower, upper), method="trf",
            max_nfev=options.max_nfev, ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        return LineFit(status="failed", message=str(exc), flags=tuple(flags+["optimizer_failed"]),
                       **common)
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return LineFit(status="failed", message=str(fitted.message), nfev=int(fitted.nfev),
                       flags=tuple(flags+["optimizer_failed"]), **common)
    if (not np.all(np.isfinite(fitted.fun)) or not np.all(np.isfinite(fitted.jac))):
        return LineFit(status="failed", message="Optimizer returned non-finite diagnostics",
                       nfev=int(fitted.nfev), flags=tuple(flags+["optimizer_failed"]), **common)
    amplitude, mu, sigma = (float(value) for value in fitted.x)
    parameters = GaussianParameters(amplitude, mu+reference, sigma)
    area = gaussian_area(amplitude, sigma)
    if area <= 0:
        return LineFit(status="failed", message="Fit has no positive absorption area",
                       nfev=int(fitted.nfev), flags=tuple(flags+["nonpositive_area"]), **common)
    unweighted_residuals = model(fitted.x)-y
    rms = float(np.sqrt(np.mean(unweighted_residuals**2)))
    chi2 = (float(np.dot(fitted.fun, fitted.fun)) / (n-3)
            if spectrum.uncertainty is not None and n > 3 else None)
    if not math.isfinite(rms) or (chi2 is not None and not math.isfinite(chi2)):
        return LineFit(status="failed", message="Non-finite residual diagnostics",
                       nfev=int(fitted.nfev), flags=tuple(flags+["optimizer_failed"]), **common)
    at_boundary = False
    for i, name in enumerate(("amplitude", "centre", "sigma")):
        tolerance = options.boundary_fraction * (upper[i]-lower[i])
        if min(fitted.x[i]-lower[i], upper[i]-fitted.x[i]) <= tolerance:
            flags.append(f"{name}_near_bound")
            at_boundary = True
    if rms > options.rms_warning:
        flags.append("high_residual_rms")
    if chi2 is not None and chi2 > options.reduced_chi2_warning:
        flags.append("high_reduced_chi2")
    displacement = parameters.centre_air_angstrom-rest
    if abs(displacement) > options.centre_warning_angstrom:
        flags.append("large_centre_displacement")
    spacing = float(np.median(np.diff(x)))
    fwhm = math.sqrt(8 * math.log(2)) * sigma
    if fwhm / spacing < options.min_samples_per_fwhm:
        flags.append("poor_fwhm_sampling")
    if np.min(np.abs(x-parameters.centre_air_angstrom)) > max(1.5*spacing, 0.5*sigma):
        flags.append("line_core_gap")
    if not np.any(x < parameters.centre_air_angstrom) or not np.any(
        x > parameters.centre_air_angstrom
    ):
        flags.append("one_sided_line_coverage")
    covariance, errors, area_error, covariance_flags = _covariance(
        fitted.jac, fitted.fun, parameters, absolute_sigma=spectrum.uncertainty is not None,
        at_boundary=at_boundary, options=options,
    )
    flags.extend(covariance_flags)
    if spectrum.uncertainty is not None and (
        (errors is not None and errors.amplitude > 0
         and amplitude / errors.amplitude < options.min_amplitude_snr)
        or amplitude < options.min_amplitude_snr * float(np.median(spectrum.uncertainty[used]))
    ):
        flags.append("low_amplitude_snr")
    return LineFit(
        status="converged", message="Numerically converged; requires scientific inspection",
        parameters=parameters, area_angstrom=area, area_error_angstrom=area_error,
        parameter_errors=errors, covariance=covariance, residual_rms=rms,
        centre_displacement_angstrom=displacement, flags=tuple(dict.fromkeys(flags)),
        nfev=int(fitted.nfev), reduced_chi2=chi2, **common,
    )


def fit_gaussian_lines(
    spectrum: Spectrum,
    *,
    windows: Mapping[str, tuple[float, float]] | None = None,
    initial_parameters: Mapping[str, GaussianParameters] | None = None,
    excluded_regions: Mapping[str, Iterable[tuple[float, float]]] | None = None,
    excluded_lines: Iterable[str] = (),
    options: FitOptions | None = None,
) -> Mapping[str, LineFit]:
    """Measure all nine Sc I regions, retaining explicit failures and missingness.

    Per-line fitting windows and exclusion regions use absolute rest-air
    Angstroms after any explicitly supplied RV correction. Initial parameters
    can be overridden per line. Otherwise a bounded starting guess is derived
    from the input pixels, without changing the fixed continuum of one.

    Fits use unweighted bounded least squares unless absolute independent pixel
    standard deviations are supplied in ``spectrum.uncertainty``. Formal errors
    condition on that model; unweighted covariance is residual-variance scaled.
    Area errors include amplitude--sigma covariance. Boundary, singular and
    insufficient-DOF cases leave covariance and formal errors unavailable.
    """
    from .spectrum import Spectrum

    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a prepared Spectrum")
    if options is None:
        options = FitOptions()
    elif not isinstance(options, FitOptions):
        raise TypeError("options must be FitOptions")
    windows = {line: _interval(value, "window", positive=True)
               for line, value in _line_mapping(windows, "windows").items()}
    initials = _line_mapping(initial_parameters, "initial_parameters")
    exclusions = {line: _regions(value, "excluded_regions")
                  for line, value in _line_mapping(excluded_regions, "excluded_regions").items()}
    if isinstance(excluded_lines, (str, bytes)):
        raise TypeError("excluded_lines must be an iterable of line IDs")
    excluded = tuple(excluded_lines)
    if (any(not isinstance(line, str) or line not in LINE_IDS for line in excluded)
            or len(set(excluded)) != len(excluded)):
        raise ValueError("excluded_lines must contain distinct IDs from line1 through line9")
    rests = get_calibration().lines
    resolved = {}
    # Validate every override before any optimization, even for unavailable lines.
    for line_id in LINE_IDS:
        rest = rests[line_id].wavelength_air_angstrom
        window = windows.get(line_id, (
            rest-options.window_half_width_angstrom, rest+options.window_half_width_angstrom,
        ))
        reference, lower, upper = _bounds(rest, window, options)
        if line_id in initials:
            _start(initials[line_id], reference, lower, upper)
        resolved[line_id] = window
    result = {}
    for line_id in LINE_IDS:
        rest, window = rests[line_id].wavelength_air_angstrom, resolved[line_id]
        if line_id in excluded:
            region, _ = fit_pixel_masks(spectrum, window)
            result[line_id] = LineFit(
                line_id=line_id, rest_wavelength_air_angstrom=rest, window=window,
                window_provisional=line_id not in windows, status="excluded",
                message="Explicitly excluded by the caller", n_excluded=int(np.sum(region)),
                excluded_regions=exclusions.get(line_id, ()),
                weighting="unweighted" if spectrum.uncertainty is None else "inverse_sigma",
            )
        else:
            result[line_id] = _fit_one(
                spectrum, line_id, rest, window, line_id not in windows,
                exclusions.get(line_id, ()), initials.get(line_id), options,
            )
    return MappingProxyType(result)
