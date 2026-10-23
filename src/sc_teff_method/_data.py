"""Read only the versioned, hash-checked numerical products shipped in this wheel."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType

from .models import Calibration, LineCalibration, SubsetPerformance

CALIBRATION_ID = "2026-10-05-nine-line-v1"
LINE_IDS = tuple(f"line{i}" for i in range(1, 10))
CONFIGURATIONS = ("nandakumar_feh", "nandakumar_no_feh")
SCOPES = ("strict_strength_domain", "all_available")
COHORTS = ("exact_subset", "common_all9")

# This anchor is filled by the reviewed, deterministic extraction tool.
_PROVENANCE_SHA256 = "f9afdda035971d77a998ed00a82cd0ea7ada23cf720026905206b1b235c40cb6"
_DATA_ROOT = files("sc_teff_method").joinpath("data", CALIBRATION_ID)


@lru_cache(maxsize=1)
def _manifest() -> dict:
    payload = _DATA_ROOT.joinpath("provenance.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != _PROVENANCE_SHA256:
        raise RuntimeError("Frozen calibration provenance failed its SHA-256 integrity check")
    return json.loads(payload)


def _read_verified(name: str) -> bytes:
    expected = _manifest()["derived_sha256"][name]
    payload = _DATA_ROOT.joinpath(name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"Frozen calibration data failed its SHA-256 integrity check: {name}")
    return payload


@lru_cache(maxsize=1)
def _calibrations() -> dict[str, Calibration]:
    data = json.loads(_read_verified("calibration.json"))
    if data["schema_version"] != 1 or data["calibration_id"] != CALIBRATION_ID:
        raise RuntimeError("Unsupported frozen calibration schema or identity")
    result = {}
    for configuration, entry in data["configurations"].items():
        lines = {}
        for line_id, values in entry["lines"].items():
            values = dict(values)
            for field in (
                "training_strength_range_angstrom",
                "reference_temperature_range_k",
                "reference_feh_range_dex",
            ):
                values[field] = tuple(values[field])
            lines[line_id] = LineCalibration(line_id=line_id, **values)
        if tuple(lines) != LINE_IDS:
            raise RuntimeError("Frozen calibration must contain exactly the nine declared lines")
        result[configuration] = Calibration(
            calibration_id=CALIBRATION_ID,
            configuration=configuration,
            model_id=entry["model_id"],
            temperature_scale_id=data["temperature_scale_id"],
            measurement_source=data["measurement_source"],
            requires_feh=entry["requires_feh"],
            declared_strength_min_angstrom=data["declared_strength_selection"]["minimum_angstrom"],
            declared_strength_max_angstrom=data["declared_strength_selection"]["maximum_angstrom"],
            fit_snapshot=entry["fit_snapshot"],
            validation_snapshot=entry["validation_snapshot"],
            lines=MappingProxyType(lines),
        )
    if tuple(result) != CONFIGURATIONS:
        raise RuntimeError("Frozen calibration configurations are incomplete")
    return result


def get_calibration(configuration: str = "nandakumar_feh") -> Calibration:
    """Return immutable coefficients, units, selection cut and empirical coverage.

    The declared strength cut is inclusive and separate from the empirical
    training-strength ranges. No network or private repository is consulted.
    """
    if not isinstance(configuration, str) or configuration not in CONFIGURATIONS:
        raise ValueError(f"configuration must be one of {CONFIGURATIONS}")
    return _calibrations()[configuration]


@lru_cache(maxsize=1)
def _subset_index() -> dict[tuple[str, str, str, int], SubsetPerformance]:
    reader = csv.DictReader(io.StringIO(_read_verified("subset_metrics.csv").decode("utf-8")))
    result = {}
    for row in reader:
        configuration, scope, cohort = (row[k] for k in ("configuration", "scope", "cohort"))
        mask = int(row["subset_mask"])
        if scope not in SCOPES or cohort not in COHORTS or not 1 <= mask <= 511:
            raise RuntimeError("Invalid frozen validation identity")
        calibration = get_calibration(configuration)
        key = (configuration, scope, cohort, mask)
        if key in result:
            raise RuntimeError("Duplicate frozen validation identity")
        result[key] = SubsetPerformance(
            line_ids=tuple(line for i, line in enumerate(LINE_IDS) if mask & (1 << i)),
            n_stars=int(row["n_stars"]),
            bias_k=float(row["bias_k"]),
            rmse_k=float(row["rmse_k"]),
            scope=scope,
            cohort=cohort,
            calibration_id=CALIBRATION_ID,
            configuration=configuration,
            model_id=calibration.model_id,
            temperature_scale_id=calibration.temperature_scale_id,
            measurement_source=calibration.measurement_source,
        )
    if len(result) != 2 * 2 * 2 * 511:
        raise RuntimeError("Frozen validation must contain all 511 subsets in each group")
    return result
