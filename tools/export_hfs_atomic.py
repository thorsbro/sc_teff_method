"""Project only explicitly permitted atomic references into compact public data.

This tool never scans directories, imports the private package or reads spectra,
star measurements or working catalogues. It reads only the fixed allowlist below.
The complete publisher Table 4 is not accessed or exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ATOMIC_ID = "hfs-atomic-v2-2026-10-02"
DATASET_ID = "hala2022_ab_generated"
SOURCE_PATHS = (
    "config/hfs_atomic_policy.json",
    "config/hfs_lines.csv",
    "config/hfs_levels.csv",
    "config/hfs_components_hala2022_ab.csv",
    "config/hfs_components_hala2022_table4.csv",
    "config/hfs_component_validation.csv",
    "src/scteff/hfs.py",
    "src/scteff/hfs_data.py",
    "src/scteff/hfs_fitting.py",
    "tests/test_hfs.py",
    "tests/test_hfs_data.py",
    "docs/hfs_atomic_data.md",
    "docs/hfs_usage.md",
)
LINE_IDS = tuple(f"line{i}" for i in range(1, 10))
COUNTS = (21, 16, 9, 12, 21, 18, 6, 12, 18)


def _rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def _packed(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def export(reference: Path, destination: Path) -> dict:
    reference, destination = reference.resolve(), destination.resolve()
    if destination.is_relative_to(reference):
        raise ValueError("Atomic export destination must be outside the read-only reference")
    for name in ("atomic.json", "atomic_provenance.json"):
        target = destination/name
        if target.is_symlink() or target.resolve().is_relative_to(reference):
            raise ValueError("Atomic export cannot replace a read-only reference through an alias")
        if target.exists() and any(target.samefile(reference/path) for path in SOURCE_PATHS):
            raise ValueError(
                "Atomic export cannot replace a read-only reference through a hard link"
            )
    sources = {name: (reference / name).read_bytes() for name in SOURCE_PATHS}
    policy = json.loads(sources["config/hfs_atomic_policy.json"])
    if policy["policy_id"] != ATOMIC_ID or policy["primary_component_dataset"] != DATASET_ID:
        raise ValueError("Unexpected frozen atomic policy or primary component dataset")
    primary_levels = [row for row in _rows(sources["config/hfs_levels.csv"])
                      if row["dataset_id"] == "hala2022_primary"]
    levels = {}
    for row in primary_levels:
        levels[row["level_id"]] = {
            "J": float(row["J"]), "A_cm1": float(row["A_model_value_cm1"]),
            "B_cm1": float(row["B_model_value_cm1"]),
            "energy_cm1": float(row["energy_cm1"]), "parity": row["parity"],
            "A_uncertainty_cm1": float(row["A_uncertainty_cm1"]),
            "B_uncertainty_cm1": float(row["B_uncertainty_cm1"])
            if row["B_uncertainty_cm1"] else None,
            "B_status": row["B_status"], "reference": row["reference"],
        }
    primary = _rows(sources["config/hfs_components_hala2022_ab.csv"])
    published = {
        (row["line_id"], row["F_lower"], row["F_upper"]): row
        for row in _rows(sources["config/hfs_components_hala2022_table4.csv"])
    }
    lines = {}
    for line in _rows(sources["config/hfs_lines.csv"]):
        key = line["line_id"]
        components = []
        for row in primary:
            if row["line_id"] != key:
                continue
            comparison = published[key, row["F_lower"], row["F_upper"]]
            components.append({
                "F_lower": float(row["F_lower"]), "F_upper": float(row["F_upper"]),
                "offset_cm1": float(row["component_offset_cm1"]),
                "strength_frozen": float(row["strength"]),
                "strength_published": float(comparison["strength_published"]),
                "table4_offset_cm1": float(comparison["component_offset_cm1"]),
            })
        lines[key] = {
            "wavelength_air_angstrom": float(line["wavelength_air_angstrom"]),
            "cog_wavenumber_cm1": float(line["hala_level_difference_cm1"]),
            "lower_level_id": line["lower_level_id"],
            "upper_level_id": line["upper_level_id"], "components": components,
        }
    if (tuple(lines) != LINE_IDS
            or tuple(len(line["components"]) for line in lines.values()) != COUNTS
            or len(levels) != 10 or len(primary) != 133 or len(published) != 133):
        raise ValueError("Unexpected nine-line atomic inventory")
    data = {
        "schema_version": 1, "atomic_id": ATOMIC_ID, "dataset_id": DATASET_ID,
        "isotope": "45Sc", "nuclear_spin_I": 3.5,
        "strength_definition": (
            "Frozen twelve-decimal production strengths projected from published Table 4 "
            "six-decimal strengths, renormalised per transition by the reference loader; "
            "neither raw strengths nor production strengths are replaced by Wigner values."
        ),
        "levels": levels, "lines": lines,
    }
    payload = _packed(data)
    manifest = {
        "schema_version": 1, "atomic_id": ATOMIC_ID, "dataset_id": DATASET_ID,
        "source_repository": "thorsbro/scteff_paper",
        "source_policy": ATOMIC_ID,
        "source_files": {
            name: {"sha256": hashlib.sha256(value).hexdigest(),
                   "git_blob_sha1": hashlib.sha1(
                       b"blob " + str(len(value)).encode() + b"\0" + value,
                   ).hexdigest()}
            for name, value in sources.items()
        },
        "publisher_table4_sha256": policy["source_files"]["hala2022_table4"]["sha256"],
        "primary_citation": "Hala & Nave (2022), ApJS 259, 17",
        "primary_doi": "10.3847/1538-4365/ac3edc",
        "atomic_data_attribution": (
            "Hala & Nave (2022), CC BY 4.0. The nine-transition selection combines "
            "their recommended A/B constants and published rounded component strengths. "
            "Lower-level recommendations cite Ertmer & Hofer (1976) and Zeiske et al. (1976)."
        ),
        "modifications": (
            "Only ten primary levels and nine canonical parent/atomic-cog mappings are "
            "exported. Components retain all 133 published F assignments, frozen A+B "
            "positions and strengths; selected Table 4 positions/raw strengths are kept "
            "only to test the A-only distinction and rounding. Approximate wavelength "
            "columns, legacy/sensitivity models and the complete publisher table are omitted."
        ),
        "derived_sha256": {"atomic.json": hashlib.sha256(payload).hexdigest()},
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "atomic.json").write_bytes(payload)
    (destination / "atomic_provenance.json").write_bytes(_packed(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = export(args.reference, args.destination)
    print(json.dumps(manifest["derived_sha256"]))


if __name__ == "__main__":
    main()
