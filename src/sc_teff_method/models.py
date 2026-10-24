"""Immutable public results and calibration metadata. Temperatures are in kelvin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

UNCERTAINTY_NOTE = (
    "Line-to-line scatter is diagnostic, not the total temperature uncertainty. "
    "Individual line errors are correlated. Held-out bias and RMSE are empirical "
    "validation statistics relative to the reference scale, not formal measurement "
    "uncertainties or a universal per-star error."
)


@dataclass(frozen=True)
class LineCalibration:
    """One frozen line calibration and its marginal empirical coverage.

    The strength range describes full-sample training measurements. Temperature
    and [Fe/H] ranges describe retained held-out references for this line.
    These ranges do not establish joint support or absolute accuracy.
    A missing metallicity coefficient is represented by None, never zero.
    """

    line_id: str
    wavelength_air_angstrom: float
    n_training: int
    a_k_per_angstrom: float
    b_k_per_dex: float | None
    c_k: float
    training_strength_range_angstrom: tuple[float, float]
    reference_temperature_range_k: tuple[float, float]
    reference_feh_range_dex: tuple[float, float]


@dataclass(frozen=True)
class Calibration:
    """The frozen numerical identity, declared selection cut and nine lines."""

    calibration_id: str
    configuration: str
    model_id: str
    temperature_scale_id: str
    measurement_source: str
    requires_feh: bool
    declared_strength_min_angstrom: float
    declared_strength_max_angstrom: float | None
    fit_snapshot: str
    validation_snapshot: str
    lines: Mapping[str, LineCalibration]


@dataclass(frozen=True)
class ApplicabilityWarning:
    """A returned diagnostic; it never removes a supplied usable measurement."""

    code: str
    line_id: str
    message: str


@dataclass(frozen=True)
class TemperatureEstimate:
    """Unweighted mean of the supplied per-line temperatures.

    ``line_scatter_k`` is the sample standard deviation (ddof=1), or None for
    one line. ``warnings`` are returned here, not emitted through Python's
    warnings module. No error bar is inferred from this result.
    """

    temperature_k: float
    line_temperatures_k: Mapping[str, float]
    used_line_ids: tuple[str, ...]
    line_scatter_k: float | None
    calibration_id: str
    configuration: str
    model_id: str
    temperature_scale_id: str
    measurement_source: str
    warnings: tuple[ApplicabilityWarning, ...]
    uncertainty_note: str = UNCERTAINTY_NOTE

    @property
    def n_lines(self) -> int:
        return len(self.used_line_ids)


@dataclass(frozen=True)
class SubsetPerformance:
    """Frozen leave-one-star-out validation for an explicit, fixed subset.

    Bias is prediction minus reference. These statistics are relative empirical
    performance, not formal measurement uncertainties for an individual target.
    """

    line_ids: tuple[str, ...]
    n_stars: int
    bias_k: float
    rmse_k: float
    scope: str
    cohort: str
    calibration_id: str
    configuration: str
    model_id: str
    temperature_scale_id: str
    measurement_source: str
    uncertainty_note: str = UNCERTAINTY_NOTE

    @property
    def n_lines(self) -> int:
        return len(self.line_ids)
