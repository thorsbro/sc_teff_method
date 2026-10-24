"""Native-pixel HFS fitting using the frozen three-parameter reference engine.

The reproduction baseline is unweighted bounded least squares. Supplied pixel
uncertainties remain available for diagnostics and manual continuum scaling;
they never silently switch this method to the Gaussian workflow's weighting.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ._data import LINE_IDS
from .continuum import ContinuumCorrection, ContinuumOptions, apply_continuum
from .fitting import (
    FitOptions,
    GaussianParameters,
    LineFit,
    _bounds,
    _coverage_flags,
    _interval,
    _line_mapping,
    _regions,
    _start,
    fit_pixel_masks,
)
from .hfs import fit_hfs_profile, wavenumber_offset_to_wavelength_offset
from .hfs_data import load_hfs_atomic_data
from .spectrum import Spectrum


@dataclass(frozen=True)
class HFSParameters(GaussianParameters):
    """Common mixture amplitude, parent centre and component sigma, in Å.

    The amplitude is the coefficient of a unit-strength component mixture,
    rather than the observed central depth of a single Gaussian.
    """


@dataclass(frozen=True)
class HFSFitOptions(FitOptions):
    """Provisional public windows/bounds with the reference HFS solver defaults.

    The default +/-3 Å windows and +/-1 Å parent-centre bounds are generic,
    user-adjustable choices, not windows copied from individual stars. The
    reference solver uses ten samples, 5000 evaluations, RMS warning 0.03,
    boundary fraction 0.001 and a 1e6 Jacobian-condition warning threshold.
    The latter is a warning, not a change to the optimiser or covariance model.
    No laboratory parameters are free fit variables.
    """

    amplitude_bounds: tuple[float, float] = (0.0, 1.0)
    min_pixels: int = 10
    max_nfev: int = 5000
    rms_warning: float = 0.03
    boundary_fraction: float = 0.001
    max_jacobian_condition: float = 1e6

    def __post_init__(self):
        super().__post_init__()
        _validate_options(self)


@dataclass(frozen=True)
class HFSLineFit(LineFit):
    """An explicitly HFS-labelled measurement or missing/failed outcome."""

    measurement_method: str = "hfs"
    atomic_dataset_id: str = "hala2022_ab_generated"
    n_components: int = 0
    component_offsets_angstrom: tuple[float, ...] = ()


def _validate_options(options):
    if not isinstance(options, FitOptions):
        raise TypeError("options must be HFSFitOptions or FitOptions")
    if options.min_pixels < 4:
        raise ValueError("HFS covariance requires at least four samples")
    if options.amplitude_bounds[1] > 1:
        raise ValueError("HFS amplitude bounds must remain in [0,1]")


def _fit_hfs_one(spectrum, line, atomic, window, provisional, regions, initial, options,
                 continuum, continuum_options):
    reference_line = atomic.lines[line]
    rest = reference_line.wavelength_air_angstrom
    components = atomic.components[line]
    offsets = wavenumber_offset_to_wavelength_offset(
        [component.offset_cm1 for component in components], rest,
        reference_line.cog_wavenumber_cm1,
    )
    working, continuum_flags = apply_continuum(
        spectrum, continuum, window=window, excluded_regions=regions,
        options=continuum_options,
    )
    region, used = fit_pixel_masks(working, window, regions)
    raw_x = working.wavelength_air_angstrom[region]
    x, y = working.wavelength_air_angstrom[used], working.flux[used]
    n = len(x)
    common = dict(
        line_id=line, rest_wavelength_air_angstrom=rest, window=window,
        window_provisional=provisional, n_pixels=n,
        n_excluded=int(np.sum(region & ~used)), excluded_regions=regions,
        weighting="unweighted", atomic_dataset_id=atomic.dataset_id,
        n_components=len(components), component_offsets_angstrom=tuple(float(v) for v in offsets),
    )
    if not len(raw_x):
        return HFSLineFit(
            status="unavailable", message="No wavelength coverage in fitting window",
            flags=(*continuum_flags, "no_coverage"), **common,
        )
    flags = [*_coverage_flags(raw_x, x, window, options), *continuum_flags]
    if n < options.min_pixels:
        return HFSLineFit(
            status="unavailable", message="Too few unmasked native pixels",
            flags=tuple([*flags, "insufficient_samples"]), **common,
        )
    conditioning_reference, lower, upper = _bounds(rest, window, options)
    if initial is None:
        eligible = (x >= lower[1]+conditioning_reference) & (x <= upper[1]+conditioning_reference)
        centre = float(x[eligible][np.argmin(y[eligible])]) if np.any(eligible) else rest
        start = np.array([max(1-float(np.min(y)), 0.01), centre-conditioning_reference, 0.35])
        epsilon = np.maximum((upper-lower)*1e-8, np.finfo(float).eps)
        start = np.clip(start, lower+epsilon, upper-epsilon)
    else:
        start = _start(initial, conditioning_reference, lower, upper)
    # The reference engine receives absolute centre bounds and applies its own
    # identical relative-centre conditioning. No data resampling is performed.
    start[1] += conditioning_reference
    lower[1] += conditioning_reference
    upper[1] += conditioning_reference
    try:
        result = fit_hfs_profile(
            x, y, components, parent_wavelength=rest,
            cog_wavenumber_cm1=reference_line.cog_wavenumber_cm1,
            initial=tuple(float(value) for value in start), bounds=(lower, upper),
            minimum_samples=options.min_pixels, max_evaluations=options.max_nfev,
            boundary_fraction=options.boundary_fraction, residual_rms_warning=options.rms_warning,
            max_jacobian_condition=options.max_jacobian_condition,
        )
    except (RuntimeError, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return HFSLineFit(status="failed", message=str(exc),
                          flags=tuple([*flags, "optimizer_failed"]), **common)
    flags.extend(result.flags)
    if result.status != "converged":
        return HFSLineFit(
            status="failed", message=result.reason,
            flags=tuple(dict.fromkeys([*flags, "optimizer_failed"])),
            nfev=result.n_evaluations, **common,
        )
    if (result.area_angstrom is None or not math.isfinite(result.area_angstrom)
            or result.area_angstrom <= 0):
        return HFSLineFit(status="failed", message="Fit has no positive HFS absorption area",
                          flags=tuple([*flags, "nonpositive_area"]),
                          nfev=result.n_evaluations, **common)
    parameters = HFSParameters(result.amplitude, result.center_angstrom, result.sigma_angstrom)
    displacement = parameters.centre_air_angstrom-rest
    if abs(displacement) > options.centre_warning_angstrom:
        flags.append("large_centre_displacement")
    spacing = float(np.median(np.diff(x)))
    if math.sqrt(8*math.log(2))*parameters.sigma_angstrom/spacing < options.min_samples_per_fwhm:
        flags.append("poor_fwhm_sampling")
    if np.min(np.abs(x-parameters.centre_air_angstrom)) > max(
        1.5*spacing, 0.5*parameters.sigma_angstrom,
    ):
        flags.append("line_core_gap")
    if not np.any(x < parameters.centre_air_angstrom) or not np.any(
        x > parameters.centre_air_angstrom
    ):
        flags.append("one_sided_line_coverage")
    component_centres = parameters.centre_air_angstrom+offsets
    if np.any((component_centres < window[0]) | (component_centres > window[1])):
        flags.append("hfs_components_cross_window")
    margin = options.boundary_fraction*(window[1]-window[0])
    if np.any(np.minimum(
        np.abs(component_centres-window[0]), np.abs(component_centres-window[1]),
    ) <= margin):
        flags.append("hfs_components_near_window_edge")
    chi2 = None
    if working.uncertainty is not None:
        from .hfs import hfs_absorption

        model = hfs_absorption(
            x, parameters.amplitude, parameters.centre_air_angstrom, parameters.sigma_angstrom,
            components, parent_wavelength=rest,
            cog_wavenumber_cm1=reference_line.cog_wavenumber_cm1,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            chi2 = float(np.sum(((model-y)/working.uncertainty[used])**2)/(n-3))
        if not math.isfinite(chi2):
            chi2 = None
            flags.append("nonfinite_chi2_diagnostic")
        elif chi2 > options.reduced_chi2_warning:
            flags.append("high_reduced_chi2")
    errors = None if result.parameter_errors is None else HFSParameters(*result.parameter_errors)
    if errors is not None and errors.amplitude > 0 and (
        parameters.amplitude/errors.amplitude < options.min_amplitude_snr
    ):
        flags.append("low_amplitude_snr")
    if any(flag.endswith("_near_bound") for flag in result.flags):
        flags.append("formal_covariance_at_boundary")
    return HFSLineFit(
        status="converged", message="HFS numerical convergence requires scientific inspection",
        parameters=parameters, area_angstrom=result.area_angstrom,
        area_error_angstrom=result.area_formal_error_angstrom, parameter_errors=errors,
        covariance=result.covariance, residual_rms=result.residual_rms,
        centre_displacement_angstrom=displacement, flags=tuple(dict.fromkeys(flags)),
        nfev=result.n_evaluations, reduced_chi2=chi2, **common,
    )


def fit_hfs_lines(
    spectrum: Spectrum,
    *,
    windows: Mapping[str, tuple[float, float]] | None = None,
    initial_parameters: Mapping[str, GaussianParameters] | None = None,
    excluded_regions: Mapping[str, Iterable[tuple[float, float]]] | None = None,
    excluded_lines: Iterable[str] = (),
    options: FitOptions | None = None,
    continuum_corrections: Mapping[str, ContinuumCorrection] | None = None,
    continuum_options: ContinuumOptions | None = None,
    weighting: str = "unweighted",
) -> Mapping[str, HFSLineFit]:
    """Fit all nine fixed HFS patterns with explicit failed/missing outcomes.

    ``spectrum`` is the existing prepared, immutable original; coordinates and
    window/mask overrides use its rest-frame air-Å grid. Each manual continuum
    correction starts from that original once. Only A, parent centre and common
    sigma are fitted. Component strengths, offsets and laboratory A/B are fixed.

    Unweighted least squares is the reproduction baseline even when input pixel
    errors are supplied. Those errors permit a separately labelled reduced-chi2
    diagnostic, but formal covariance remains residual-scaled unweighted and
    includes amplitude-width covariance. Weighted HFS fitting is not implemented.
    Numerical convergence is never scientific acceptance.
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a prepared Spectrum")
    if weighting != "unweighted":
        raise ValueError("Only weighting='unweighted' is validated for HFS reproduction")
    options = HFSFitOptions() if options is None else options
    _validate_options(options)
    continuum_options = ContinuumOptions() if continuum_options is None else continuum_options
    if not isinstance(continuum_options, ContinuumOptions):
        raise TypeError("continuum_options must be ContinuumOptions")
    windows = {line: _interval(value, "window", positive=True)
               for line, value in _line_mapping(windows, "windows").items()}
    initials = _line_mapping(initial_parameters, "initial_parameters")
    exclusions = {line: _regions(value, "excluded_regions")
                  for line, value in _line_mapping(excluded_regions, "excluded_regions").items()}
    corrections = _line_mapping(continuum_corrections, "continuum_corrections")
    if any(not isinstance(value, ContinuumCorrection) for value in corrections.values()):
        raise TypeError("Each continuum correction must be ContinuumCorrection")
    if isinstance(excluded_lines, (str, bytes)):
        raise TypeError("excluded_lines must be an iterable of line IDs")
    excluded = tuple(excluded_lines)
    if (any(not isinstance(line, str) or line not in LINE_IDS for line in excluded)
            or len(set(excluded)) != len(excluded)):
        raise ValueError("excluded_lines must contain distinct IDs from line1 through line9")
    atomic = load_hfs_atomic_data()
    resolved = {}
    for line in LINE_IDS:
        rest = atomic.lines[line].wavelength_air_angstrom
        window = windows.get(line, (
            rest-options.window_half_width_angstrom, rest+options.window_half_width_angstrom,
        ))
        reference, lower, upper = _bounds(rest, window, options)
        if line in initials:
            _start(initials[line], reference, lower, upper)
        # Validate local continuum controls before any optimisation, including
        # deliberately excluded or unavailable lines; never hide an invalid C.
        correction = corrections.get(line, ContinuumCorrection())
        apply_continuum(spectrum, correction, window=window, options=continuum_options)
        resolved[line] = window
    fits = {}
    for line in LINE_IDS:
        rest, window = atomic.lines[line].wavelength_air_angstrom, resolved[line]
        regions = exclusions.get(line, ())
        if line in excluded:
            region, _ = fit_pixel_masks(spectrum, window)
            offsets = wavenumber_offset_to_wavelength_offset(
                [component.offset_cm1 for component in atomic.components[line]], rest,
                atomic.lines[line].cog_wavenumber_cm1,
            )
            fits[line] = HFSLineFit(
                line_id=line, rest_wavelength_air_angstrom=rest, window=window,
                window_provisional=line not in windows, status="excluded",
                message="Explicitly excluded by the caller", n_excluded=int(np.sum(region)),
                excluded_regions=regions, weighting="unweighted",
                atomic_dataset_id=atomic.dataset_id, n_components=len(offsets),
                component_offsets_angstrom=tuple(float(value) for value in offsets),
            )
        else:
            fits[line] = _fit_hfs_one(
                spectrum, line, atomic, window, line not in windows, regions, initials.get(line),
                options, corrections.get(line, ContinuumCorrection()), continuum_options,
            )
    return MappingProxyType(fits)
