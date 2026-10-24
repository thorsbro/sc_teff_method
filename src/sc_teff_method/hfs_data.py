"""Hash-checked nine-line atomic data with fixed A+B component positions.

Only the adopted production pattern is selectable for fitting. The selected
Table 4 offsets/rounded strengths are retained as a documented numerical check,
never silently substituted for the production positions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType

from ._data import LINE_IDS, get_calibration
from .hfs import (
    HFSComponent,
    HFSLevel,
    _normalize_strengths,
    generate_hfs_components,
    hyperfine_energy_shift,
)

ATOMIC_ID = "hfs-atomic-v2-2026-10-02"
HFS_DATASET_ID = "hala2022_ab_generated"
COMPONENT_COUNTS = MappingProxyType(dict(zip(
    LINE_IDS, (21, 16, 9, 12, 21, 18, 6, 12, 18), strict=True,
)))
_PROVENANCE_SHA256 = "e5662ea0c5c6b0be4aa6fe8cd9cd6d2ba1472a4db9073a9315be039617cf8af4"
_DATA_ROOT = files("sc_teff_method").joinpath("data", ATOMIC_ID)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class HFSAtomicPolicy:
    policy_id: str
    I: float  # noqa: E741
    primary_level_dataset: str
    primary_component_dataset: str
    raw: Mapping


@dataclass(frozen=True)
class HFSLine:
    """Canonical air parent and separately supplied atomic transition wavenumber."""

    line_id: str
    wavelength_air_angstrom: float
    cog_wavenumber_cm1: float
    lower_level_id: str
    upper_level_id: str


@dataclass(frozen=True)
class HFSLevelRecord:
    """Primary laboratory constant and explicit quadrupole-reporting status."""

    dataset_id: str
    level_id: str
    level: HFSLevel
    energy_cm1: float
    B_status: str
    source: Mapping


@dataclass(frozen=True)
class HFSAtomicData:
    dataset_id: str
    policy: HFSAtomicPolicy
    lines: Mapping[str, HFSLine]
    levels: Mapping[str, HFSLevelRecord]
    components: Mapping[str, tuple[HFSComponent, ...]]
    published_components: Mapping[str, tuple[HFSComponent, ...]]
    raw_strengths: Mapping[str, tuple[float, ...]]
    source_hashes: Mapping[str, str]
    provenance: Mapping


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"Invalid frozen atomic value: {name}")
    return float(value)


def _normalised_components(records, *, published=False):
    strengths = [row["strength_published" if published else "strength_frozen"] for row in records]
    weights = _normalize_strengths(strengths)
    # Preserve the frozen reference loader's final Python-sum normalisation.
    weights[-1] = 1.0 - sum(float(value) for value in weights[:-1])
    return tuple(
        HFSComponent(row["F_lower"], row["F_upper"],
                     row["table4_offset_cm1" if published else "offset_cm1"], float(weight))
        for row, weight in zip(records, weights, strict=True)
    )


def _decode_atomic(payload, provenance):
    if (
        payload["schema_version"] != 1 or payload["atomic_id"] != ATOMIC_ID
        or payload["dataset_id"] != HFS_DATASET_ID or payload["isotope"] != "45Sc"
        or payload["nuclear_spin_I"] != 3.5 or tuple(payload["lines"]) != LINE_IDS
    ):
        raise RuntimeError("Unsupported frozen HFS atomic identity")
    levels = {}
    for key, row in payload["levels"].items():
        level = HFSLevel(row["J"], row["A_cm1"], row["B_cm1"])
        status = row["B_status"]
        if status not in {"measured", "measured_recommended", "not_applicable_J_lt_1",
                          "unreported_assumed_zero"}:
            raise RuntimeError("Unsupported frozen quadrupole status")
        if (level.J < 1) != (status == "not_applicable_J_lt_1"):
            raise RuntimeError("Invalid physically inapplicable quadrupole status")
        if status in {"not_applicable_J_lt_1", "unreported_assumed_zero"} and (
            level.B != 0 or row["B_uncertainty_cm1"] is not None
        ):
            raise RuntimeError("Invalid unreported/inapplicable B=0 assumption")
        if row["parity"] not in {"even", "odd"}:
            raise RuntimeError("Invalid frozen level parity")
        for name in ("A_uncertainty_cm1", "B_uncertainty_cm1"):
            if row[name] is not None and _number(row[name], name) < 0:
                raise RuntimeError("Negative frozen laboratory uncertainty")
        levels[key] = HFSLevelRecord("hala2022_primary", key, level,
                                     _number(row["energy_cm1"], "energy_cm1"), status, _freeze(row))
    required = {row[name] for row in payload["lines"].values()
                for name in ("lower_level_id", "upper_level_id")}
    if set(levels) != required or len(levels) != 10:
        raise RuntimeError("Frozen electronic level inventory differs from nine transitions")
    if levels["upper_2D5_2"].level.A != 0.02205 or levels["upper_2D5_2"].level.B != -0.01581:
        raise RuntimeError("Frozen line9 requires the positive-A assignment")
    lines, components, comparison, raw = {}, {}, {}, {}
    canonical = get_calibration().lines
    for key, row in payload["lines"].items():
        line = HFSLine(key, _number(row["wavelength_air_angstrom"], "air wavelength"),
                       _number(row["cog_wavenumber_cm1"], "atomic wavenumber"),
                       row["lower_level_id"], row["upper_level_id"])
        if (line.wavelength_air_angstrom != canonical[key].wavelength_air_angstrom
                or line.cog_wavenumber_cm1 <= 0):
            raise RuntimeError("Frozen HFS parent wavelength or atomic wavenumber changed")
        lower, upper = levels[line.lower_level_id], levels[line.upper_level_id]
        if lower.source["parity"] != "even" or upper.source["parity"] != "odd" or not math.isclose(
            upper.energy_cm1-lower.energy_cm1, line.cog_wavenumber_cm1, abs_tol=1e-8, rel_tol=0,
        ):
            raise RuntimeError("Frozen HFS transition energy/parity mismatch")
        records = row["components"]
        if len(records) != COMPONENT_COUNTS[key]:
            raise RuntimeError("Frozen HFS component count changed")
        for entry in records:
            for name, value in entry.items():
                _number(value, name)
        primary = _normalised_components(records)
        published = _normalised_components(records, published=True)
        theoretical = {
            (c.F_lower, c.F_upper): c
            for c in generate_hfs_components(3.5, lower.level, upper.level)
        }
        pairs = {(c.F_lower, c.F_upper) for c in primary}
        if len(pairs) != len(records) or pairs != set(theoretical):
            raise RuntimeError("Frozen HFS component assignments/selection rules changed")
        expected_sum = 1.000001 if key == "line1" else 0.999999 if key == "line5" else 1.0
        if not math.isclose(sum(r["strength_published"] for r in records), expected_sum,
                            rel_tol=0, abs_tol=1e-14):
            raise RuntimeError("Frozen Table 4 rounded strength sum changed")
        if abs(math.fsum(r["strength_frozen"] for r in records)-1) > 2e-11:
            raise RuntimeError("Frozen production strength sum exceeds twelve-decimal rounding")
        for entry, component, pub in zip(records, primary, published, strict=True):
            theory = theoretical[component.F_lower, component.F_upper]
            if (abs(component.offset_cm1-theory.offset_cm1) > 5.01e-13
                    or abs(entry["strength_published"]-theory.strength) > 5.01e-7
                    or abs(component.strength-pub.strength) > 2e-12):
                raise RuntimeError("Frozen HFS A+B or published-strength regression failed")
            a_only = hyperfine_energy_shift(3.5, upper.level.J, component.F_upper, upper.level.A)
            a_only -= hyperfine_energy_shift(3.5, lower.level.J, component.F_lower, lower.level.A)
            if abs(pub.offset_cm1-a_only) > 2.37e-5:
                raise RuntimeError("Frozen Table 4 positions no longer match A-only regression")
        lines[key], components[key], comparison[key] = line, primary, published
        raw[key] = tuple(entry["strength_published"] for entry in records)
    policy = HFSAtomicPolicy(ATOMIC_ID, 3.5, "hala2022_primary", HFS_DATASET_ID, _freeze({
        "strength_definition": payload["strength_definition"],
        "line9_assignment": "positive A=0.02205 cm^-1; B=-0.01581 cm^-1",
        "missing_B": "Unreported B=0 is a modelling assumption, distinct from J<1 inapplicability.",
    }))
    hashes = {name: values["sha256"] for name, values in provenance["source_files"].items()}
    return HFSAtomicData(HFS_DATASET_ID, policy, MappingProxyType(lines), MappingProxyType(levels),
                         MappingProxyType(components), MappingProxyType(comparison),
                         MappingProxyType(raw), MappingProxyType(hashes), _freeze(provenance))


@lru_cache(maxsize=1)
def load_hfs_atomic_data(dataset: str = HFS_DATASET_ID) -> HFSAtomicData:
    """Read the adopted fixed pattern from this installation, with no private access.

    The production loader normalises the frozen twelve-decimal strengths exactly
    as the reference. Raw published rounded strengths and Table 4 positions are
    separate comparison metadata. Alternate atomic datasets are not fitting modes.
    """
    if dataset != HFS_DATASET_ID:
        raise ValueError(f"The supported fixed HFS dataset is {HFS_DATASET_ID}")
    raw_manifest = _DATA_ROOT.joinpath("atomic_provenance.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != _PROVENANCE_SHA256:
        raise RuntimeError("Frozen HFS atomic provenance failed SHA-256 integrity")
    provenance = json.loads(raw_manifest)
    raw = _DATA_ROOT.joinpath("atomic.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["derived_sha256"]["atomic.json"]:
        raise RuntimeError("Frozen HFS atomic data failed SHA-256 integrity")
    return _decode_atomic(json.loads(raw), provenance)
