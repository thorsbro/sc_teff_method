"""Apply frozen affine calibrations to caller-supplied Gaussian areas."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from statistics import mean, stdev
from types import MappingProxyType

from ._data import COHORTS, LINE_IDS, SCOPES, _subset_index, get_calibration
from .models import ApplicabilityWarning, SubsetPerformance, TemperatureEstimate


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number (not a boolean or string)")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _canonical_lines(line_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(line_ids, (str, bytes)):
        raise TypeError("line_ids must be an iterable of line IDs, not a single string")
    try:
        selected = tuple(line_ids)
    except TypeError as exc:
        raise TypeError("line_ids must be an iterable of line IDs") from exc
    if not selected:
        raise ValueError("Supply a non-empty subset of line1 through line9")
    if any(not isinstance(line, str) or line not in LINE_IDS for line in selected):
        raise ValueError("Unknown line ID; use line1 through line9")
    if len(set(selected)) != len(selected):
        raise ValueError("Line IDs must be distinct")
    return tuple(line for line in LINE_IDS if line in selected)


def estimate_temperature(
    areas: Mapping[str, Real],
    *,
    configuration: str = "nandakumar_feh",
    feh: Real | None = None,
    reference_temperature_k: Real | None = None,
) -> TemperatureEstimate:
    """Estimate Teff from any non-empty subset of nine Gaussian areas in Angstrom.

    Areas must be positive finite real numbers. Omit unavailable lines; None,
    zero, blank strings and NaN are not missing-value conventions. Each supplied
    line is retained, including extrapolations, and receives its own temperature.
    The result averages those temperatures with equal weights.

    ``nandakumar_feh`` requires finite [Fe/H] in dex. ``nandakumar_no_feh`` uses
    its independent metallicity-free fit; optional ``feh`` only checks reference
    coverage in that configuration and never enters its prediction.

    The inclusive declared cut W >= 0.08 Angstrom and empirical training ranges
    produce separate returned warnings, without clipping or excluding inputs.
    Temperature coverage is checked against an optional external reference;
    otherwise each predicted line temperature serves only as an applicability
    proxy. Metallicity and temperature coverage are marginal, not joint support.
    """
    calibration = get_calibration(configuration)
    if not isinstance(areas, Mapping):
        raise TypeError("areas must be a mapping from line IDs to Gaussian areas in Angstrom")
    chosen = _canonical_lines(areas)
    if calibration.requires_feh and feh is None:
        raise ValueError("nandakumar_feh requires an explicit finite feh ([Fe/H] in dex)")
    metal = None if feh is None else _finite_real(feh, "feh")
    reference = (
        None if reference_temperature_k is None
        else _finite_real(reference_temperature_k, "reference_temperature_k")
    )
    if reference is not None and reference <= 0:
        raise ValueError("reference_temperature_k must be positive")
    predictions, warnings = {}, []
    for line_id in chosen:
        line = calibration.lines[line_id]
        area = _finite_real(areas[line_id], f"{line_id} area")
        if area <= 0:
            raise ValueError(f"{line_id} area must be positive; omit missing lines")
        # Keep the reference implementation's arithmetic order.
        temperature = line.a_k_per_angstrom * area + line.c_k
        if line.b_k_per_dex is not None:
            temperature += line.b_k_per_dex * metal
        if not math.isfinite(temperature):
            raise ValueError(f"{line_id} area or feh produces a non-finite temperature")
        predictions[line_id] = temperature

        def warn(code: str, message: str) -> None:
            warnings.append(ApplicabilityWarning(code, line_id, f"{line_id}: {message}"))

        if area < calibration.declared_strength_min_angstrom:
            warn(
                "outside_declared_strength_cut",
                f"area {area:g} Angstrom is below the inclusive declared selection cut "
                f"of {calibration.declared_strength_min_angstrom:g} Angstrom; "
                "strict_strength_domain validation does not cover this measurement",
            )
        lower, upper = line.training_strength_range_angstrom
        if not lower <= area <= upper:
            warn(
                "outside_training_strength_range",
                f"area {area:g} Angstrom is outside the empirical training strength range "
                f"[{lower:g}, {upper:g}] Angstrom",
            )
        lower, upper = line.reference_temperature_range_k
        check_temperature = temperature if reference is None else reference
        if not lower <= check_temperature <= upper:
            origin = "estimated line temperature (applicability proxy)" if reference is None else (
                "supplied reference temperature"
            )
            warn(
                "outside_reference_temperature_range",
                f"{origin} {check_temperature:g} K is outside empirical reference-temperature "
                f"coverage [{lower:g}, {upper:g}] K",
            )
        lower, upper = line.reference_feh_range_dex
        if metal is not None and not lower <= metal <= upper:
            warn(
                "outside_reference_feh_range",
                f"[Fe/H] {metal:g} dex is outside empirical reference-metallicity coverage "
                f"[{lower:g}, {upper:g}] dex",
            )

    return TemperatureEstimate(
        temperature_k=mean(predictions.values()),
        line_temperatures_k=MappingProxyType(predictions),
        used_line_ids=chosen,
        line_scatter_k=stdev(predictions.values()) if len(predictions) > 1 else None,
        calibration_id=calibration.calibration_id,
        configuration=configuration,
        model_id=calibration.model_id,
        temperature_scale_id=calibration.temperature_scale_id,
        measurement_source=calibration.measurement_source,
        warnings=tuple(warnings),
    )


def subset_performance(
    line_ids: Iterable[str],
    *,
    configuration: str = "nandakumar_feh",
    scope: str = "strict_strength_domain",
    cohort: str = "exact_subset",
) -> SubsetPerformance:
    """Look up frozen held-out N, bias and RMSE for an exact fixed line identity.

    ``strict_strength_domain`` requires the declared Gaussian-anchor strength
    selection cut; ``all_available`` retains finite held-out predictions without
    that cut. Neither scope asserts empirical training-range or joint support.

    ``exact_subset`` uses every reference star with all requested lines usable.
    ``common_all9`` uses the same complete-nine-line reference cohort for every
    subset in its model and scope. Missing lines are never replaced by zero.
    Scope and cohort select validation rows; they do not filter target areas.
    """
    get_calibration(configuration)
    if not isinstance(scope, str) or scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if not isinstance(cohort, str) or cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}")
    chosen = _canonical_lines(line_ids)
    mask = sum(1 << LINE_IDS.index(line) for line in chosen)
    return _subset_index()[configuration, scope, cohort, mask]
