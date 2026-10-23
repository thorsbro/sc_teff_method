"""Connect Gaussian fits to the unchanged frozen temperature/validation API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ._data import LINE_IDS, SCOPES, get_calibration
from .api import _finite_real, estimate_temperature, subset_performance
from .continuum import ContinuumCorrection
from .fitting import FitOptions, GaussianParameters, LineFit, fit_gaussian_lines
from .models import ApplicabilityWarning, SubsetPerformance, TemperatureEstimate
from .spectrum import Spectrum, prepare_spectrum

QUALITY_POLICIES = ("exclude_warned", "include_warned")


@dataclass(frozen=True)
class SpectralEstimate:
    """All nine fit outcomes and the explicitly included temperature subset.

    A converged fit is a numerical result requiring scientific inspection.
    ``line_temperatures_k`` includes all converged positive-area fits for
    diagnostics, including quality-flagged fits omitted from ``temperature``.
    Missing/failed/user-excluded lines have no area or diagnostic temperature.
    ``continuum_corrections`` optionally records manually chosen per-line
    preprocessing over the original ``spectrum`` so downstream plots/exports
    can reconstruct the corrected flux actually fitted. PR 2 results use None.
    """

    spectrum: Spectrum
    fits: Mapping[str, LineFit]
    line_temperatures_k: Mapping[str, float]
    calibration_warnings: tuple[ApplicabilityWarning, ...]
    included_line_ids: tuple[str, ...]
    exclusion_reasons: Mapping[str, str]
    temperature: TemperatureEstimate | None
    performance: SubsetPerformance | None
    configuration: str
    feh: float | None
    reference_temperature_k: float | None
    quality_policy: str
    validation_scope: str
    fit_options: FitOptions
    initial_parameters: Mapping[str, GaussianParameters]
    target_passes_declared_strength_cut: bool | None
    warnings: tuple[str, ...]
    continuum_corrections: Mapping[str, ContinuumCorrection] | None = None


def estimate_from_spectrum(
    spectrum: Spectrum,
    *,
    configuration: str = "nandakumar_feh",
    feh: float | None = None,
    reference_temperature_k: float | None = None,
    windows: Mapping[str, tuple[float, float]] | None = None,
    initial_parameters: Mapping[str, GaussianParameters] | None = None,
    excluded_regions: Mapping[str, Iterable[tuple[float, float]]] | None = None,
    excluded_lines: Iterable[str] = (),
    options: FitOptions | None = None,
    quality_policy: str = "exclude_warned",
    validation_scope: str = "strict_strength_domain",
) -> SpectralEstimate:
    """Fit all nine lines, then average the chosen positive, converged areas' Teff.

    ``exclude_warned`` (default) omits every converged fit carrying quality
    flags. ``include_warned`` explicitly permits those fits after inspection;
    it never includes failed/unavailable fits or user-excluded lines.
    Provisional windows and the need for scientific inspection apply to all
    fits, independently of this numerical quality policy.
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a Spectrum from prepare_spectrum or load_spectrum")
    calibration = get_calibration(configuration)
    if calibration.requires_feh and feh is None:
        raise ValueError("nandakumar_feh requires an explicit finite feh ([Fe/H] in dex)")
    metal = None if feh is None else _finite_real(feh, "feh")
    reference = (None if reference_temperature_k is None else
                 _finite_real(reference_temperature_k, "reference_temperature_k"))
    if reference is not None and reference <= 0:
        raise ValueError("reference_temperature_k must be positive")
    if not isinstance(quality_policy, str) or quality_policy not in QUALITY_POLICIES:
        raise ValueError(f"quality_policy must be one of {QUALITY_POLICIES}")
    if not isinstance(validation_scope, str) or validation_scope not in SCOPES:
        raise ValueError(f"validation_scope must be one of {SCOPES}")
    fit_options = FitOptions() if options is None else options
    fitted = fit_gaussian_lines(
        spectrum, windows=windows, initial_parameters=initial_parameters,
        excluded_regions=excluded_regions, excluded_lines=excluded_lines, options=fit_options,
    )
    usable = {
        line: fit.area_angstrom for line, fit in fitted.items()
        if fit.status == "converged" and fit.area_angstrom is not None and fit.area_angstrom > 0
    }
    diagnostic = (estimate_temperature(
        usable, configuration=configuration, feh=metal, reference_temperature_k=reference,
    ) if usable else None)
    included, excluded = {}, {}
    for line in LINE_IDS:
        fit = fitted[line]
        if line not in usable:
            excluded[line] = f"{fit.status}: {fit.message}"
        elif fit.flags and quality_policy == "exclude_warned":
            excluded[line] = "quality flags: " + ", ".join(fit.flags)
        else:
            included[line] = usable[line]
    temperature = (estimate_temperature(
        included, configuration=configuration, feh=metal, reference_temperature_k=reference,
    ) if included else None)
    performance = (subset_performance(
        included, configuration=configuration, scope=validation_scope, cohort="exact_subset",
    ) if included else None)
    warnings = [
        "Inspect all line fits before scientific use; numerical convergence is not acceptance.",
        "Synthetic numerical tests do not establish accuracy for new instruments or populations.",
    ]
    if any(fit.window_provisional for fit in fitted.values()):
        warnings.append(
            "Default symmetric fitting windows are provisional and may need adjustment."
        )
    if spectrum.uncertainty is not None:
        warnings.append(
            "Supplied pixel uncertainties enable inverse-sigma weighting. The frozen calibration "
            "used unweighted measurements; transfer of weighted measurements is not validated."
        )
    if any(line in included and fitted[line].flags for line in fitted):
        warnings.append("Quality-flagged fits were included by the explicit include_warned policy.")
    passes_cut = (all(area >= calibration.declared_strength_min_angstrom
                      for area in included.values()) if included else None)
    if passes_cut is False and validation_scope == "strict_strength_domain":
        warnings.append(
            "At least one included area fails the declared strength cut; strict-domain "
            "validation statistics do not cover this measurement."
        )
    if not included:
        warnings.append(
            "No usable lines remain under the selected inclusion policy; Teff is missing."
        )
    return SpectralEstimate(
        spectrum=spectrum, fits=MappingProxyType(dict(fitted)),
        line_temperatures_k=MappingProxyType(
            dict(diagnostic.line_temperatures_k) if diagnostic else {}
        ),
        calibration_warnings=diagnostic.warnings if diagnostic else (),
        included_line_ids=tuple(included), exclusion_reasons=MappingProxyType(excluded),
        temperature=temperature, performance=performance,
        configuration=configuration, feh=metal, reference_temperature_k=reference,
        quality_policy=quality_policy, validation_scope=validation_scope,
        fit_options=fit_options,
        initial_parameters=MappingProxyType(dict(initial_parameters or {})),
        target_passes_declared_strength_cut=passes_cut,
        warnings=tuple(warnings),
    )


def fit_spectrum(
    wavelength,
    flux,
    *,
    wavelength_unit: str,
    wavelength_frame: str,
    continuum_normalised: bool,
    uncertainty=None,
    mask=None,
    rv_kms: float | None = None,
    configuration: str = "nandakumar_feh",
    feh: float | None = None,
    reference_temperature_k: float | None = None,
    windows: Mapping[str, tuple[float, float]] | None = None,
    initial_parameters: Mapping[str, GaussianParameters] | None = None,
    excluded_regions: Mapping[str, Iterable[tuple[float, float]]] | None = None,
    excluded_lines: Iterable[str] = (),
    options: FitOptions | None = None,
    quality_policy: str = "exclude_warned",
    validation_scope: str = "strict_strength_domain",
) -> SpectralEstimate:
    """Array interface; explicitly declare air Angstroms and a prepared continuum.

    Positive ``rv_kms`` denotes recession/redshift. Rest wavelengths are
    input wavelengths divided by sqrt((1+v/c)/(1-v/c)); flux is not resampled.
    Windows, initial centres and exclusions always use corrected rest air Å.
    """
    spectrum = prepare_spectrum(
        wavelength, flux, wavelength_unit=wavelength_unit, wavelength_frame=wavelength_frame,
        continuum_normalised=continuum_normalised, uncertainty=uncertainty, mask=mask,
        rv_kms=rv_kms,
    )
    return estimate_from_spectrum(
        spectrum, configuration=configuration, feh=feh,
        reference_temperature_k=reference_temperature_k, windows=windows,
        initial_parameters=initial_parameters, excluded_regions=excluded_regions,
        excluded_lines=excluded_lines, options=options,
        quality_policy=quality_policy, validation_scope=validation_scope,
    )
