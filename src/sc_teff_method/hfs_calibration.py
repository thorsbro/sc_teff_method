"""Frozen HFS-specific temperature calibration and Gaussian-anchor validation.

These separately named functions never use Gaussian coefficients. The frozen
model IDs are shared across methods, so measurement_source='hfs' is essential to
their identity. This numerical API needs only Python's standard library.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from numbers import Real
from statistics import mean, stdev
from types import MappingProxyType

from ._data import CALIBRATION_ID, COHORTS, CONFIGURATIONS, LINE_IDS, SCOPES
from .api import _canonical_lines, _finite_real
from .models import (
    ApplicabilityWarning,
    Calibration,
    LineCalibration,
    SubsetPerformance,
    TemperatureEstimate,
)

_PROVENANCE_SHA256 = "f838b0eadba36ff5d26df2edc291a719424bc1acd635ca5b970929e0a1e68833"
_DATA_ROOT = files("sc_teff_method").joinpath("data", CALIBRATION_ID + "-hfs")
HFS_SELECTION_NOTE = (
    "Strict-strength-domain validation used Gaussian-anchor W >= 0.08 Angstrom, "
    "not HFS areas. A lookup alone does not establish target applicability. Frozen "
    "statistics do not validate new continuum choices or post-hoc line selection."
)


@dataclass(frozen=True)
class HFSCalibration(Calibration):
    """HFS coefficients; inherited declared strength limits concern Gaussian anchors."""

    selection_anchor_source: str = "gaussian"
    atomic_component_dataset: str = "hala2022_ab_generated"
    selection_note: str = HFS_SELECTION_NOTE


@dataclass(frozen=True)
class HFSTemperatureEstimate(TemperatureEstimate):
    """HFS mean with an explicit three-state Gaussian-anchor domain check.

    True requires a passing Gaussian anchor for every supplied HFS line; False
    means at least one known anchor fails; None means required anchors are
    missing and no known failure establishes exclusion. This is only the declared
    selection cut, not empirical training coverage or validated instrument transfer.
    """

    strict_strength_domain_applicable: bool | None = None
    gaussian_anchor_areas: Mapping[str, float] | None = None
    selection_note: str = HFS_SELECTION_NOTE


@dataclass(frozen=True)
class HFSSubsetPerformance(SubsetPerformance):
    """Frozen HFS statistics; strict-domain rows were selected using Gaussian anchors."""

    selection_anchor_source: str = "gaussian"
    selection_note: str = HFS_SELECTION_NOTE


@lru_cache(maxsize=1)
def _manifest() -> dict:
    payload = _DATA_ROOT.joinpath("provenance.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != _PROVENANCE_SHA256:
        raise RuntimeError("Frozen HFS provenance failed its SHA-256 integrity check")
    manifest = json.loads(payload)
    if (manifest["calibration_id"] != CALIBRATION_ID
            or manifest["measurement_source"] != "hfs"):
        raise RuntimeError("Unexpected HFS calibration provenance identity")
    return manifest


def _read_verified(name: str) -> bytes:
    expected = _manifest()["derived_sha256"][name]
    payload = _DATA_ROOT.joinpath(name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"Frozen HFS calibration failed its SHA-256 integrity check: {name}")
    return payload


@lru_cache(maxsize=1)
def _calibrations() -> dict[str, HFSCalibration]:
    data = json.loads(_read_verified("calibration.json"))
    selection = data["declared_strength_selection"]
    if (data["schema_version"] != 1 or data["calibration_id"] != CALIBRATION_ID
            or data["measurement_source"] != "hfs"
            or data["atomic_component_dataset"] != "hala2022_ab_generated"
            or selection["measurement_source"] != "gaussian"
            or selection["minimum_angstrom"] != 0.08
            or selection["maximum_angstrom"] is not None or selection["inclusive"] is not True):
        raise RuntimeError("Unsupported frozen HFS calibration identity/selection")
    calibrations = {}
    for configuration, entry in data["configurations"].items():
        lines = {}
        for line_id, record in entry["lines"].items():
            record = dict(record)
            for field in (
                "training_strength_range_angstrom", "reference_temperature_range_k",
                "reference_feh_range_dex",
            ):
                record[field] = tuple(record[field])
            lines[line_id] = LineCalibration(line_id=line_id, **record)
        if tuple(lines) != LINE_IDS:
            raise RuntimeError("Frozen HFS calibration needs exactly the nine declared lines")
        calibrations[configuration] = HFSCalibration(
            calibration_id=CALIBRATION_ID,
            configuration=configuration,
            model_id=entry["model_id"],
            temperature_scale_id=data["temperature_scale_id"],
            measurement_source="hfs",
            requires_feh=entry["requires_feh"],
            declared_strength_min_angstrom=selection["minimum_angstrom"],
            declared_strength_max_angstrom=selection["maximum_angstrom"],
            fit_snapshot=entry["fit_snapshot"],
            validation_snapshot=entry["validation_snapshot"],
            lines=MappingProxyType(lines),
        )
    if tuple(calibrations) != CONFIGURATIONS:
        raise RuntimeError("Frozen HFS calibration configurations are incomplete")
    return calibrations


def get_hfs_calibration(configuration: str = "nandakumar_feh") -> HFSCalibration:
    """Return immutable HFS coefficients, HFS training ranges and anchor metadata."""
    if not isinstance(configuration, str) or configuration not in CONFIGURATIONS:
        raise ValueError(f"configuration must be one of {CONFIGURATIONS}")
    return _calibrations()[configuration]


def _validate_anchors(anchors) -> dict[str, float] | None:
    if anchors is None:
        return None
    if not isinstance(anchors, Mapping):
        raise TypeError("gaussian_anchor_areas must be a mapping of Gaussian areas in Angstrom")
    if getattr(anchors, "measurement_source", "gaussian") != "gaussian":
        raise ValueError("Gaussian-anchor selection requires Gaussian areas, not HFS areas")
    if any(not isinstance(line, str) or line not in LINE_IDS for line in anchors):
        raise ValueError("Unknown Gaussian-anchor line ID; use line1 through line9")
    checked = {}
    for line in LINE_IDS:
        if line in anchors:
            area = _finite_real(anchors[line], f"{line} Gaussian anchor area")
            if area <= 0:
                raise ValueError("Gaussian anchor areas must be positive; omit missing anchors")
            checked[line] = area
    return checked


def estimate_hfs_temperature(
    areas: Mapping[str, Real],
    *,
    configuration: str = "nandakumar_feh",
    feh: Real | None = None,
    reference_temperature_k: Real | None = None,
    gaussian_anchor_areas: Mapping[str, Real] | None = None,
) -> HFSTemperatureEstimate:
    """Average per-line HFS temperatures with the frozen HFS-specific coefficients.

    Supply positive HFS areas; missing lines must be omitted. Method-tagged
    Gaussian mappings are rejected. Plain mappings are explicitly declared HFS
    by choosing this function. Gaussian anchors are optional separate measurements
    for checking the shared paper selection; they never enter HFS predictions.
    Their absence warns that strict-domain applicability cannot be established.
    """
    calibration = get_hfs_calibration(configuration)
    if not isinstance(areas, Mapping):
        raise TypeError("areas must be a mapping of HFS areas in Angstrom")
    if getattr(areas, "measurement_source", "hfs") != "hfs":
        raise ValueError("HFS calibration requires HFS areas; Gaussian areas need the Gaussian API")
    chosen = _canonical_lines(areas)
    if calibration.requires_feh and feh is None:
        raise ValueError("nandakumar_feh requires an explicit finite feh ([Fe/H] in dex)")
    metal = None if feh is None else _finite_real(feh, "feh")
    reference = (None if reference_temperature_k is None else
                 _finite_real(reference_temperature_k, "reference_temperature_k"))
    if reference is not None and reference <= 0:
        raise ValueError("reference_temperature_k must be positive")
    anchors = _validate_anchors(gaussian_anchor_areas)
    predictions, warnings = {}, []
    known_failure = False
    unknown_anchor = False
    for line_id in chosen:
        line = calibration.lines[line_id]
        area = _finite_real(areas[line_id], f"{line_id} HFS area")
        if area <= 0:
            raise ValueError("HFS areas must be positive; omit missing lines")
        temperature = line.a_k_per_angstrom * area + line.c_k
        if line.b_k_per_dex is not None:
            temperature += line.b_k_per_dex * metal
        if not math.isfinite(temperature):
            raise ValueError(f"{line_id} HFS area or feh produces a non-finite temperature")
        predictions[line_id] = temperature

        def warn(code, message):
            warnings.append(ApplicabilityWarning(code, line_id, f"{line_id}: {message}"))

        if anchors is None or line_id not in anchors:
            unknown_anchor = True
            warn(
                "strict_strength_domain_unknown",
                "Gaussian anchor unavailable; strict-strength-domain applicability is unknown. "
                "The HFS area must not be substituted for the Gaussian selection anchor.",
            )
        elif anchors[line_id] < calibration.declared_strength_min_angstrom:
            known_failure = True
            warn(
                "outside_gaussian_anchor_strength_domain",
                f"Gaussian anchor {anchors[line_id]:g} Angstrom is below the inclusive "
                "0.08 Angstrom selection cut; strict-domain statistics do not cover this target.",
            )
        lo, hi = line.training_strength_range_angstrom
        if not lo <= area <= hi:
            warn("outside_training_strength_range",
                 f"HFS area {area:g} Angstrom is outside empirical HFS training range "
                 f"[{lo:g}, {hi:g}] Angstrom")
        lo, hi = line.reference_temperature_range_k
        value = temperature if reference is None else reference
        if not lo <= value <= hi:
            origin = "estimated line temperature (applicability proxy)" if reference is None else (
                "supplied reference temperature"
            )
            warn("outside_reference_temperature_range",
                 f"{origin} {value:g} K is outside empirical reference-temperature coverage "
                 f"[{lo:g}, {hi:g}] K")
        lo, hi = line.reference_feh_range_dex
        if metal is not None and not lo <= metal <= hi:
            warn("outside_reference_feh_range",
                 f"[Fe/H] {metal:g} dex is outside empirical reference-metallicity coverage "
                 f"[{lo:g}, {hi:g}] dex")
    applicability = False if known_failure else (None if unknown_anchor else True)
    return HFSTemperatureEstimate(
        temperature_k=mean(predictions.values()),
        line_temperatures_k=MappingProxyType(predictions),
        used_line_ids=chosen,
        line_scatter_k=stdev(predictions.values()) if len(predictions) > 1 else None,
        calibration_id=CALIBRATION_ID,
        configuration=configuration,
        model_id=calibration.model_id,
        temperature_scale_id=calibration.temperature_scale_id,
        measurement_source="hfs",
        warnings=tuple(warnings),
        strict_strength_domain_applicable=applicability,
        gaussian_anchor_areas=None if anchors is None else MappingProxyType(anchors),
    )


@lru_cache(maxsize=1)
def _subset_index() -> dict[tuple[str, str, str, int], HFSSubsetPerformance]:
    reader = csv.DictReader(io.StringIO(_read_verified("subset_metrics.csv").decode("utf-8")))
    result = {}
    for row in reader:
        configuration, scope, cohort = (
            row[field] for field in ("configuration", "scope", "cohort")
        )
        mask = int(row["subset_mask"])
        if scope not in SCOPES or cohort not in COHORTS or not 1 <= mask <= 511:
            raise RuntimeError("Invalid frozen HFS validation identity")
        calibration = get_hfs_calibration(configuration)
        key = configuration, scope, cohort, mask
        if key in result:
            raise RuntimeError("Duplicate frozen HFS subset identity")
        result[key] = HFSSubsetPerformance(
            line_ids=tuple(line for i, line in enumerate(LINE_IDS) if mask & (1 << i)),
            n_stars=int(row["n_stars"]), bias_k=float(row["bias_k"]), rmse_k=float(row["rmse_k"]),
            scope=scope, cohort=cohort, calibration_id=CALIBRATION_ID,
            configuration=configuration, model_id=calibration.model_id,
            temperature_scale_id=calibration.temperature_scale_id, measurement_source="hfs",
        )
    expected = {(name, scope, cohort, mask) for name in CONFIGURATIONS for scope in SCOPES
                for cohort in COHORTS for mask in range(1, 512)}
    if set(result) != expected:
        raise RuntimeError("Frozen HFS validation requires every subset in every supported group")
    return result


def hfs_subset_performance(
    line_ids: Iterable[str],
    *,
    configuration: str = "nandakumar_feh",
    scope: str = "strict_strength_domain",
    cohort: str = "exact_subset",
) -> HFSSubsetPerformance:
    """Look up HFS held-out N/bias/RMSE for a fixed subset, scope and cohort.

    Lookup does not establish applicability to a target. Strict-domain rows were
    selected using Gaussian anchors; supplying HFS areas cannot test that cut.
    ``all_available`` removes that selection cut but does not guarantee empirical
    training coverage, scientific fit quality or instrument compatibility.
    """
    get_hfs_calibration(configuration)
    if not isinstance(scope, str) or scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if not isinstance(cohort, str) or cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}")
    chosen = _canonical_lines(line_ids)
    mask = sum(1 << LINE_IDS.index(line) for line in chosen)
    return _subset_index()[configuration, scope, cohort, mask]
