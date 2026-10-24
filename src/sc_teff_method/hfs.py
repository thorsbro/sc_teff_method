"""Reusable first-order HFS physics and fixed-continuum profile fitting.

Adapted from the frozen reference numerical implementation. It needs only
Python, NumPy and SciPy, without repository or spectrum dependencies. Angular
momenta are integer or half-integer numbers. Component offsets are wavenumbers in cm^-1; wavelengths
and fitted widths are Angstroms. A/B energy units are otherwise caller-selected.

References: Hala & Nave (2022), ApJS 259, 17, doi:10.3847/1538-4365/ac3edc;
the Racah formula for the Wigner 6-j symbol, NIST DLMF 34.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


def _real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _angular(value: float, name: str) -> int:
    """Validate an angular momentum and return twice its integer value."""
    try:
        doubled = 2 * _real(value, name)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite angular momentum") from error
    if not math.isfinite(doubled) or doubled < 0 or not doubled.is_integer():
        raise ValueError(f"{name} must be a finite nonnegative integer or half-integer")
    return int(doubled)


@dataclass(frozen=True)
class HFSLevel:
    """Electronic level: J, magnetic dipole A and electric quadrupole B.

    A and B must use the same units, e.g. cm^-1 or MHz. B defaults to zero as a
    caller-selected assumption, not a claim that an unmeasured B is zero.
    """

    J: float
    A: float
    B: float = 0.0

    def __post_init__(self) -> None:
        _angular(self.J, "J")
        for name in ("J", "A", "B"):
            object.__setattr__(self, name, _real(getattr(self, name), name))


@dataclass(frozen=True)
class HFSComponent:
    """One component with F assignments, offset in cm^-1 and positive strength.

    Strengths are normalized per transition by generation and fitting. Offset is
    E_upper(F_upper) - E_lower(F_lower), relative to the unsplit transition.
    """

    F_lower: float
    F_upper: float
    offset_cm1: float
    strength: float

    def __post_init__(self) -> None:
        lower, upper = _angular(self.F_lower, "F_lower"), _angular(self.F_upper, "F_upper")
        if abs(lower - upper) not in (0, 2) or (lower == 0 and upper == 0):
            raise ValueError("HFS components require Delta F=0,+/-1 and forbid 0 -> 0")
        for name in ("F_lower", "F_upper", "offset_cm1", "strength"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if self.strength <= 0:
            raise ValueError("Component strength must be positive")


def allowed_f_values(I: float, J: float) -> tuple[float, ...]:  # noqa: E741
    """Return F=|I-J|, |I-J|+1, ..., I+J for integer/half-integer I and J."""
    i, j = _angular(I, "I"), _angular(J, "J")
    return tuple(f / 2 for f in range(abs(i - j), i + j + 1, 2))


def allowed_hfs_transitions(
    I: float,  # noqa: E741
    J_lower: float,
    J_upper: float,
) -> tuple[tuple[float, float], ...]:
    """Return (F_lower,F_upper) pairs satisfying electric-dipole selection rules.

    Delta F=0,+/-1; 0 -> 0 is forbidden. Electronic Delta J=0,+/-1 also applies,
    with J=0 -> J=0 forbidden. A rule-allowed pair can have zero Wigner strength.
    """
    lower, upper = allowed_f_values(I, J_lower), allowed_f_values(I, J_upper)
    if abs(J_lower - J_upper) > 1 or J_lower + J_upper < 1 or int(2 * (J_lower + J_upper)) % 2:
        return ()
    return tuple(
        (fl, fu) for fl in lower for fu in upper if abs(fl - fu) <= 1 and (fl != 0 or fu != 0)
    )


def hyperfine_energy_shift(
    I: float,  # noqa: E741
    J: float,
    F: float,
    A: float,
    B: float = 0.0,
) -> float:
    """Return the first-order Casimir HFS shift, in the same units as A and B.

    K=F(F+1)-I(I+1)-J(J+1). E=A*K/2 + B*[3*K*(K+1)/4
    - I*(I+1)*J*(J+1)]/[2*I*(2*I-1)*J*(2*J-1)]. The quadrupole term
    is omitted for I<1 or J<1. No MHz-to-cm^-1 conversion is implicit.
    """
    _angular(F, "F")
    if F not in allowed_f_values(I, J):
        raise ValueError("F is not an allowed value for I and J")
    A, B = _real(A, "A"), _real(B, "B")
    k = F * (F + 1) - I * (I + 1) - J * (J + 1)
    shift = 0.5 * A * k
    if I >= 1 and J >= 1:
        shift += (
            B
            * (0.75 * k * (k + 1) - I * (I + 1) * J * (J + 1))
            / (2 * I * (2 * I - 1) * J * (2 * J - 1))
        )
    return float(shift)


def _wigner_6j_squared(a: int, b: int, c: int, d: int, e: int, f: int) -> Fraction:
    """Exact Racah sum squared for doubled integer/half-integer arguments.

    Squaring avoids intermediate square roots; Fraction/factorial arithmetic
    prevents overflow and cancellation in the finite sum. Only its final
    nonnegative squared value is converted to float by the caller.
    """
    triangles = ((a, b, c), (a, e, f), (d, b, f), (d, e, c))
    delta_squared = Fraction(1)
    for x, y, z in triangles:
        if (x + y + z) % 2 or abs(x - y) > z or x + y < z:
            return Fraction(0)
        delta_squared *= Fraction(
            math.factorial((x + y - z) // 2)
            * math.factorial((x - y + z) // 2)
            * math.factorial((-x + y + z) // 2),
            math.factorial((x + y + z) // 2 + 1),
        )
    alpha = tuple(sum(t) // 2 for t in triangles)
    beta = ((a + b + d + e) // 2, (a + c + d + f) // 2, (b + c + e + f) // 2)
    total = Fraction(0)
    for z in range(max(alpha), min(beta) + 1):
        denominator = math.prod(math.factorial(z - x) for x in alpha)
        denominator *= math.prod(math.factorial(x - z) for x in beta)
        total += Fraction((-1) ** z * math.factorial(z + 1), denominator)
    return delta_squared * total**2


def hfs_relative_strength(
    I: float,  # noqa: E741
    J_lower: float,
    J_upper: float,
    F_lower: float,
    F_upper: float,
) -> float:
    """Return an unnormalized electric-dipole component strength.

    S=(2*F_lower+1)*(2*F_upper+1)*{J_upper F_upper I;
    F_lower J_lower 1}^2. Normalize over the whole transition for a profile.
    Valid but forbidden/zero-strength pairs return zero; invalid F values raise.
    """
    i, jl, ju, fl, fu = (
        _angular(v, n)
        for v, n in (
            (I, "I"),
            (J_lower, "J_lower"),
            (J_upper, "J_upper"),
            (F_lower, "F_lower"),
            (F_upper, "F_upper"),
        )
    )
    if F_lower not in allowed_f_values(I, J_lower) or F_upper not in allowed_f_values(I, J_upper):
        raise ValueError("F assignments do not match I and electronic J")
    if (F_lower, F_upper) not in allowed_hfs_transitions(I, J_lower, J_upper):
        return 0.0
    return float((fl + 1) * (fu + 1) * _wigner_6j_squared(ju, fu, i, fl, jl, 2))


def _normalize_strengths(strengths: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(strengths, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("A component pattern needs finite one-dimensional strengths")
    if np.any(values <= 0) or not np.isfinite(np.sum(values)):
        raise ValueError("Component strengths must be positive with a finite sum")
    values = values / np.sum(values)
    # Correct only floating-point summation error, after explicit normalization.
    index = int(np.argmax(values))
    for _ in range(2):
        values[index] += 1.0 - np.sum(values)
    return values


def generate_hfs_components(
    I: float,  # noqa: E741
    lower: HFSLevel,
    upper: HFSLevel,
) -> tuple[HFSComponent, ...]:
    """Generate positive-strength HFS components from caller-supplied cm^-1 A/B.

    Positions are upper minus lower HFS energy shifts. Zero Wigner-strength
    components are omitted. Return strengths normalized to unit sum. This
    function does not fit laboratory constants or access project configuration.
    """
    pairs = allowed_hfs_transitions(I, lower.J, upper.J)
    positive = [(fl, fu, hfs_relative_strength(I, lower.J, upper.J, fl, fu)) for fl, fu in pairs]
    positive = [(fl, fu, s) for fl, fu, s in positive if s > 0]
    weights = _normalize_strengths([s for _, _, s in positive])
    weights[-1] = 1.0 - sum(float(s) for s in weights[:-1])
    return tuple(
        HFSComponent(
            fl,
            fu,
            hyperfine_energy_shift(I, upper.J, fu, upper.A, upper.B)
            - hyperfine_energy_shift(I, lower.J, fl, lower.A, lower.B),
            float(weight),
        )
        for (fl, fu, _), weight in zip(positive, weights, strict=True)
    )


def wavenumber_offset_to_wavelength_offset(
    offset_cm1: ArrayLike, parent_wavelength: float, cog_wavenumber_cm1: float
) -> NDArray[np.float64]:
    """Derive wavelength offsets in Angstroms from cm^-1 component offsets.

    lambda_i=lambda0*sigma0/(sigma0+delta_sigma); delta_lambda=lambda_i-lambda0.
    Parent lambda0 may be air wavelength: only the relative pattern is applied.
    sigma0 is the atomic centre-of-gravity wavenumber, not 1e8/lambda_air.
    Positive wavenumber offsets give negative wavelength offsets.
    """
    values = np.asarray(offset_cm1)
    if values.dtype.kind not in "iuf":
        raise ValueError("Wavenumber offsets must be finite real values")
    offsets = np.asarray(values, dtype=float)
    parent_wavelength = _real(parent_wavelength, "parent_wavelength")
    cog_wavenumber_cm1 = _real(cog_wavenumber_cm1, "cog_wavenumber_cm1")
    if (
        not np.isfinite(parent_wavelength)
        or parent_wavelength <= 0
        or not np.isfinite(cog_wavenumber_cm1)
        or cog_wavenumber_cm1 <= 0
        or np.any(~np.isfinite(offsets))
        or np.any(~np.isfinite(cog_wavenumber_cm1 + offsets))
        or np.any(cog_wavenumber_cm1 + offsets <= 0)
    ):
        raise ValueError("Wavelength and component wavenumbers must be finite and positive")
    # Algebraically the exact ratio above, evaluated without subtracting two
    # large nearby wavelengths; offset zero consequently remains exactly zero.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = -parent_wavelength * offsets / (cog_wavenumber_cm1 + offsets)
    if np.any(~np.isfinite(result)):
        raise ValueError("Wavenumber conversion produced a nonfinite wavelength offset")
    return result


def _pattern(
    components: tuple[HFSComponent, ...], parent_wavelength: float, cog_wavenumber_cm1: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        components = tuple(components)
    except TypeError as exc:
        raise ValueError("A pattern requires HFSComponent entries") from exc
    if not components or any(not isinstance(component, HFSComponent) for component in components):
        raise ValueError("A pattern requires non-empty HFSComponent entries")
    offsets = wavenumber_offset_to_wavelength_offset(
        [c.offset_cm1 for c in components], parent_wavelength, cog_wavenumber_cm1
    )
    return offsets, _normalize_strengths([c.strength for c in components])


def _profile(
    x: NDArray, amplitude: float, center: float, sigma: float, offsets: NDArray, strengths: NDArray
) -> NDArray[np.float64]:
    d = x[..., None] - center - offsets
    exponential = np.exp(-(d**2) / (2 * sigma**2))
    # Only guard summation roundoff: a positive unit-strength Gaussian mixture
    # is mathematically in [0,1], including coincident components at their peaks.
    kernel = np.clip(exponential @ strengths, 0.0, 1.0)
    return 1.0 - amplitude * kernel


def hfs_absorption(
    wavelength: ArrayLike,
    amplitude: float,
    center: float,
    sigma: float,
    components: tuple[HFSComponent, ...],
    *,
    parent_wavelength: float,
    cog_wavenumber_cm1: float,
) -> NDArray[np.float64]:
    """Evaluate 1-A*sum(s_i*exp[-(lambda-center-offset_i)^2/(2*sigma^2)]).

    Continuum is fixed at one; sigma is one shared Gaussian width in Angstroms.
    Strengths are normalized. Amplitude must lie in [0,1], ensuring nonnegative
    flux despite overlap. Center is the fitted parent position; relative offsets
    use the supplied fixed parent wavelength and atomic sigma0. This empirical
    profile is not radiative-transfer synthesis and has no saturation parameter.
    """
    amplitude, center, sigma = (
        _real(value, name) for value, name in
        ((amplitude, "amplitude"), (center, "center"), (sigma, "sigma"))
    )
    if not 0 <= amplitude <= 1:
        raise ValueError("Amplitude must be finite and in [0,1]")
    if not np.isfinite(center) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Center must be finite and sigma finite and positive")
    x = np.asarray(wavelength)
    if x.dtype.kind not in "iuf" or np.any(~np.isfinite(x)):
        raise ValueError("Wavelengths must be finite real values")
    offsets, weights = _pattern(components, parent_wavelength, cog_wavenumber_cm1)
    return _profile(x.astype(float, copy=False), amplitude, center, sigma, offsets, weights)


def hfs_area(amplitude: float, sigma: float) -> float:
    """Full analytic area A*sigma*sqrt(2*pi), in Angstroms for Angstrom sigma."""
    amplitude, sigma = _real(amplitude, "amplitude"), _real(sigma, "sigma")
    if (
        not np.isfinite(amplitude)
        or not 0 <= amplitude <= 1
        or not np.isfinite(sigma)
        or sigma <= 0
    ):
        raise ValueError("Amplitude must be in [0,1] and sigma finite and positive")
    area = float(amplitude * sigma * np.sqrt(2 * np.pi))
    if not math.isfinite(area):
        raise ValueError("HFS area must be finite")
    return area


@dataclass(frozen=True)
class HFSFit:
    """Numerical fit result; convergence is never scientific approval.

    Areas/centers/sigmas use Angstroms. Formal error is residual-scaled unweighted
    covariance including amplitude-sigma covariance. It excludes continuum,
    correlated noise, blends and model systematics. Unsuccessful values are None.
    """

    status: str
    reason: str
    flags: tuple[str, ...]
    n_samples: int
    amplitude: float | None = None
    center_angstrom: float | None = None
    sigma_angstrom: float | None = None
    area_angstrom: float | None = None
    area_formal_error_angstrom: float | None = None
    residual_rms: float | None = None
    n_evaluations: int = 0
    covariance: tuple[tuple[float, ...], ...] | None = None
    parameter_errors: tuple[float, float, float] | None = None


def _formal_diagnostics(
    jacobian: NDArray, residuals: NDArray, amplitude: float, sigma: float, n_samples: int,
    *, max_jacobian_condition: float = 1e6,
) -> tuple[float | None, tuple[str, ...], NDArray | None]:
    """Shared Gaussian/HFS residual-scaled SVD covariance, retaining A-sigma covariance."""
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    error = None
    covariance = None
    flags = []
    if singular[-1] <= np.finfo(float).eps * max(jacobian.shape) * singular[0]:
        flags.append("singular_covariance")
    else:
        if singular[0] / singular[-1] > max_jacobian_condition:
            flags.append("ill_conditioned_covariance")
        covariance = ((vt.T / singular**2) @ vt) * np.sum(residuals**2) / (n_samples - 3)
        gradient = np.sqrt(2 * np.pi) * np.array([sigma, 0, amplitude])
        variance = float(gradient @ covariance @ gradient)
        if np.isfinite(variance) and variance >= 0:
            error = float(np.sqrt(variance))
        else:
            flags.append("invalid_covariance")
    return error, tuple(flags), covariance


def _area_formal_error(
    jacobian: NDArray, residuals: NDArray, amplitude: float, sigma: float, n_samples: int,
    *, max_jacobian_condition: float = 1e6,
) -> tuple[float | None, tuple[str, ...]]:
    """Reference-compatible formal area-error helper, retaining A-sigma covariance."""
    error, flags, _ = _formal_diagnostics(
        jacobian, residuals, amplitude, sigma, n_samples,
        max_jacobian_condition=max_jacobian_condition,
    )
    return error, flags


def fit_hfs_profile(
    wavelength: ArrayLike,
    flux: ArrayLike,
    components: tuple[HFSComponent, ...],
    *,
    parent_wavelength: float,
    cog_wavenumber_cm1: float,
    initial: tuple[float, float, float] | None = None,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    minimum_samples: int = 10,
    max_evaluations: int = 5000,
    boundary_fraction: float = 0.001,
    residual_rms_warning: float = 0.03,
    max_jacobian_condition: float = 1e6,
) -> HFSFit:
    """Fit caller-owned wavelength/flux arrays with three free parameters A,center,sigma.

    Arrays must be one-dimensional, equal-length, with finite increasing wavelengths.
    No masks, continuum corrections or wavelength transforms are implicit. Flux
    must already have continuum one. Initial/bounds triples use (A,center,sigma).
    Defaults estimate depth/center/width and bound A in [0,1], center to supplied
    samples and sigma to [span*1e-6,span]. Bounds may restrict but never extend A
    outside [0,1]. Invalid configuration raises ValueError; insufficient/nonfinite
    flux returns an explicit status. Fits use unweighted bounded least squares,
    an analytic Jacobian and relative-center conditioning. No laboratory A/B
    constants, strengths, offsets or continuum parameters are fitted.
    """
    x, y = np.asarray(wavelength, dtype=float), np.asarray(flux, dtype=float)
    boundary_fraction = _real(boundary_fraction, "boundary_fraction")
    residual_rms_warning = _real(residual_rms_warning, "residual_rms_warning")
    max_jacobian_condition = _real(max_jacobian_condition, "max_jacobian_condition")
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Fit inputs must be equal one-dimensional arrays")
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("Fit wavelengths must be finite and increasing")
    offsets, weights = _pattern(components, parent_wavelength, cog_wavenumber_cm1)
    if (
        type(minimum_samples) is not int
        or minimum_samples < 4
        or type(max_evaluations) is not int
        or max_evaluations < 1
        or not np.isfinite(boundary_fraction)
        or not 0 < boundary_fraction < 0.5
        or not np.isfinite(residual_rms_warning)
        or residual_rms_warning <= 0
        or max_jacobian_condition <= 1
    ):
        raise ValueError("Invalid fitting sample/evaluation limits or diagnostics")
    n = len(x)
    if n < minimum_samples:
        return HFSFit("insufficient_samples", "Too few native samples in fit window", (), n)
    if not np.all(np.isfinite(y)):
        return HFSFit("invalid_samples", "Nonfinite flux in fit window", (), n)
    reference = (x[0] + x[-1]) / 2
    span = x[-1] - x[0]
    if bounds is None:
        lower = np.array([0.0, x[0], max(span * 1e-6, np.finfo(float).eps)])
        upper = np.array([1.0, x[-1], span])
    else:
        lower, upper = (np.array(v, dtype=float, copy=True) for v in bounds)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or lower[0] < 0
        or upper[0] > 1
        or lower[2] <= 0
    ):
        raise ValueError("Bounds must be finite ordered (amplitude,center,sigma) triples")
    if initial is None:
        guess = np.array([max(0.01, 1 - np.min(y)), x[np.argmin(y)], span / 6])
        guess = np.clip(guess, lower, upper)
    else:
        guess = np.array(initial, dtype=float)
    if (
        guess.shape != (3,)
        or not np.all(np.isfinite(guess))
        or np.any(guess < lower)
        or np.any(guess > upper)
    ):
        raise ValueError("Initial guesses outside fit bounds")
    # Use the bounds midpoint, matching the existing project's window scaling.
    reference = (lower[1] + upper[1]) / 2
    x = x - reference
    lower[1] -= reference
    upper[1] -= reference
    guess[1] -= reference

    def residual(parameters):
        return _profile(x, *parameters, offsets, weights) - y

    def jacobian(parameters):
        amplitude, mu, sigma = parameters
        d = x[:, None] - mu - offsets
        exponential = np.exp(-(d**2) / (2 * sigma**2))
        return np.column_stack(
            (
                -(exponential @ weights),
                -amplitude * ((exponential * d) @ weights) / sigma**2,
                -amplitude * ((exponential * d**2) @ weights) / sigma**3,
            )
        )

    try:
        result = least_squares(
            residual,
            guess,
            jac=jacobian,
            bounds=(lower, upper),
            max_nfev=max_evaluations,
            method="trf",
        )
    except (RuntimeError, FloatingPointError) as error:
        return HFSFit("fit_failed", str(error), (), n)
    if not result.success or not np.all(np.isfinite(result.x)):
        return HFSFit("fit_failed", result.message, (), n, n_evaluations=result.nfev)
    amplitude, center, sigma = result.x
    rms = float(np.sqrt(np.mean(result.fun**2)))
    flags = []
    for i, name in enumerate(("amplitude", "center", "sigma")):
        tolerance = boundary_fraction * (upper[i] - lower[i])
        if min(result.x[i] - lower[i], upper[i] - result.x[i]) <= tolerance:
            flags.append(f"{name}_near_bound")
    if rms > residual_rms_warning:
        flags.append("high_residual_rms")
    error, covariance_flags, covariance = _formal_diagnostics(
        result.jac, result.fun, amplitude, sigma, n,
        max_jacobian_condition=max_jacobian_condition,
    )
    flags.extend(covariance_flags)
    errors = None
    if covariance is not None and np.all(np.diag(covariance) >= 0):
        errors = tuple(float(v) for v in np.sqrt(np.diag(covariance)))
    return HFSFit(
        "converged",
        "",
        tuple(flags),
        n,
        float(amplitude),
        float(center + reference),
        float(sigma),
        hfs_area(amplitude, sigma),
        error,
        rms,
        result.nfev,
        (tuple(tuple(float(v) for v in row) for row in covariance)
         if covariance is not None else None),
        errors,
    )
