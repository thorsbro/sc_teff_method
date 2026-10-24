"""Explicit HFS measurements connected only to the frozen HFS calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ._data import LINE_IDS, SCOPES
from .api import _finite_real
from .continuum import ContinuumCorrection, ContinuumOptions, apply_continuum
from .hfs_calibration import (
    estimate_hfs_temperature,
    get_hfs_calibration,
    hfs_subset_performance,
)
from .hfs_fitting import HFSFitOptions, HFSLineFit, fit_hfs_lines
from .measurements import LineAreas
from .models import ApplicabilityWarning, SubsetPerformance, TemperatureEstimate
from .spectrum import Spectrum, prepare_spectrum
from .workflow import QUALITY_POLICIES


@dataclass(frozen=True)
class HFSSpectralEstimate:
    """HFS-specific result, deliberately distinct from a Gaussian estimate.

    The source spectrum remains immutable. Reconstruct the working input with
    ``corrected_spectrum(line_id)``; each correction starts from that source.
    A numerical inclusion policy never constitutes scientific approval.
    """

    spectrum: Spectrum
    fits: Mapping[str, HFSLineFit]
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
    fit_options: HFSFitOptions
    initial_parameters: Mapping
    continuum_corrections: Mapping[str, ContinuumCorrection]
    continuum_options: ContinuumOptions
    gaussian_anchor_areas: Mapping[str, float] | None
    strict_strength_domain_applicable: bool | None
    warnings: tuple[str, ...]
    measurement_source: str = "hfs"
    atomic_dataset_id: str = "hala2022_ab_generated"

    @property
    def areas(self) -> LineAreas:
        """Included HFS areas with a method label for calibration safety."""
        return LineAreas(
            {line: self.fits[line].area_angstrom for line in self.included_line_ids}, "hfs",
        )

    def corrected_spectrum(self, line_id: str) -> Spectrum:
        if line_id not in LINE_IDS:
            raise ValueError("Unknown line ID; use line1 through line9")
        spectrum, _ = apply_continuum(
            self.spectrum, self.continuum_corrections.get(line_id, ContinuumCorrection()),
            window=self.fits[line_id].window, excluded_regions=self.fits[line_id].excluded_regions,
            options=self.continuum_options,
        )
        return spectrum


def estimate_hfs_from_spectrum(
    spectrum: Spectrum,
    *,
    configuration="nandakumar_feh",
    feh=None,
    reference_temperature_k=None,
    windows=None,
    initial_parameters=None,
    excluded_regions=None,
    excluded_lines=(),
    options=None,
    quality_policy="exclude_warned",
    validation_scope="strict_strength_domain",
    gaussian_anchor_areas=None,
    continuum_corrections=None,
    continuum_options=None,
    weighting="unweighted",
) -> HFSSpectralEstimate:
    """Fit HFS patterns and average the included individual HFS temperatures.

    Unweighted least squares is the reference measurement procedure. Supplied
    uncertainties are retained/transformed but do not enable weighting. Strict
    validation uses separately supplied Gaussian anchors, never HFS area cuts.
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a prepared Spectrum")
    calibration = get_hfs_calibration(configuration)
    if calibration.requires_feh and feh is None:
        raise ValueError("nandakumar_feh requires explicit [Fe/H]")
    metal = None if feh is None else _finite_real(feh, "feh")
    reference = (None if reference_temperature_k is None else
                 _finite_real(reference_temperature_k, "reference_temperature_k"))
    if reference is not None and reference <= 0:
        raise ValueError("reference_temperature_k must be positive")
    if quality_policy not in QUALITY_POLICIES:
        raise ValueError(f"quality_policy must be one of {QUALITY_POLICIES}")
    if validation_scope not in SCOPES:
        raise ValueError(f"validation_scope must be one of {SCOPES}")
    anchors = None
    if gaussian_anchor_areas is not None:
        if not isinstance(gaussian_anchor_areas, Mapping):
            raise TypeError("gaussian_anchor_areas must be a mapping")
        if getattr(gaussian_anchor_areas, "measurement_source", "gaussian") != "gaussian":
            raise ValueError("Strict-domain selection requires Gaussian anchors, not HFS areas")
        anchors = {}
        for line, value in gaussian_anchor_areas.items():
            if line not in LINE_IDS:
                raise ValueError("Unknown Gaussian anchor line ID")
            area = _finite_real(value, f"{line} Gaussian anchor")
            if area <= 0:
                raise ValueError("Gaussian anchor areas must be positive; omit missing anchors")
            anchors[line] = area
    fit_options = HFSFitOptions() if options is None else options
    correction_options = ContinuumOptions() if continuum_options is None else continuum_options
    fits = fit_hfs_lines(
        spectrum, windows=windows, initial_parameters=initial_parameters,
        excluded_regions=excluded_regions, excluded_lines=excluded_lines, options=fit_options,
        continuum_corrections=continuum_corrections, continuum_options=correction_options,
        weighting=weighting,
    )
    usable = {line: fit.area_angstrom for line, fit in fits.items()
              if fit.status == "converged" and fit.area_angstrom is not None
              and fit.area_angstrom > 0}
    included, excluded = {}, {}
    for line, fit in fits.items():
        if line not in usable:
            excluded[line] = f"{fit.status}: {fit.message}"
        elif fit.flags and quality_policy == "exclude_warned":
            excluded[line] = "quality flags: " + ", ".join(fit.flags)
        else:
            included[line] = usable[line]
    arguments = dict(configuration=configuration, feh=metal, reference_temperature_k=reference,
                     gaussian_anchor_areas=anchors)
    diagnostic = estimate_hfs_temperature(LineAreas(usable, "hfs"), **arguments) if usable else None
    temperature = (estimate_hfs_temperature(LineAreas(included, "hfs"), **arguments)
                   if included else None)
    performance = (hfs_subset_performance(
        included, configuration=configuration, scope=validation_scope, cohort="exact_subset",
    ) if included else None)
    applicability = (temperature.strict_strength_domain_applicable if temperature else None)
    warnings = [
        "Inspect HFS profiles and residuals before scientific use; convergence is not acceptance.",
        "The full HFS area uses fixed A+B components and HFS-specific calibration coefficients.",
        "Formal Gaussian-mixture covariance excludes continuum, profile and calibration "
        "systematics; empirical RMSE is not a formal per-target temperature uncertainty.",
        "Frozen subset statistics do not validate post-hoc selection, newly chosen continuum "
        "corrections or transfer to arbitrary instrumental conditions.",
        "Analytical synthetic validation establishes numerical correctness, not observational "
        "accuracy, saturation physics or reliable transfer to other instruments.",
    ]
    if validation_scope == "strict_strength_domain" and applicability is not True:
        warnings.append(
            "Strict-domain target applicability is "
            + ("unknown without a complete passing set of Gaussian anchors. "
               if applicability is None else "false because a Gaussian anchor fails the cut. ")
            + "The displayed frozen validation row does not establish target applicability; "
            "HFS areas are never used as Gaussian selection anchors."
        )
    if spectrum.uncertainty is not None:
        warnings.append(
            "Pixel uncertainties were supplied and are transformed with any manual continuum, "
            "but fitting remains explicitly unweighted to reproduce the reference procedure."
        )
    if any(control.active for control in (continuum_corrections or {}).values()):
        warnings.append(
            "Local continuum controls are manual preprocessing, not the calibration paper's "
            "continuum procedure. Placement uncertainty is excluded from formal fit errors."
        )
    if any(fit.window_provisional for fit in fits.values()):
        warnings.append(
            "Default symmetric HFS windows are provisional and may truncate components."
        )
    if any(fits[line].flags for line in included):
        warnings.append("Flagged HFS fits were included by the explicit include_warned policy.")
    if not included:
        warnings.append(
            "No usable HFS lines remain; temperature and subset performance are missing."
        )
    return HFSSpectralEstimate(
        spectrum=spectrum, fits=MappingProxyType(dict(fits)),
        line_temperatures_k=MappingProxyType(dict(diagnostic.line_temperatures_k)
                                           if diagnostic else {}),
        calibration_warnings=diagnostic.warnings if diagnostic else (),
        included_line_ids=tuple(included), exclusion_reasons=MappingProxyType(excluded),
        temperature=temperature, performance=performance, configuration=configuration, feh=metal,
        reference_temperature_k=reference, quality_policy=quality_policy,
        validation_scope=validation_scope, fit_options=fit_options,
        initial_parameters=MappingProxyType(dict(initial_parameters or {})),
        continuum_corrections=MappingProxyType(dict(continuum_corrections or {})),
        continuum_options=correction_options,
        gaussian_anchor_areas=None if anchors is None else MappingProxyType(anchors),
        strict_strength_domain_applicable=applicability, warnings=tuple(warnings),
    )


def fit_hfs_spectrum(
    wavelength, flux, *, wavelength_unit, wavelength_frame, continuum_normalised,
    uncertainty=None, mask=None, rv_kms=None, **kwargs,
) -> HFSSpectralEstimate:
    """Array workflow with the same explicit preparation/RV convention as PR 2."""
    spectrum = prepare_spectrum(
        wavelength, flux, wavelength_unit=wavelength_unit, wavelength_frame=wavelength_frame,
        continuum_normalised=continuum_normalised,
        uncertainty=uncertainty, mask=mask, rv_kms=rv_kms,
    )
    return estimate_hfs_from_spectrum(spectrum, **kwargs)
