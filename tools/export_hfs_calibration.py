"""Project only approved frozen aggregate HFS products; never refit a calibration.

This maintainer command reads exactly four private files, all explicitly listed
below. It never imports the private package, opens per-star products or walks a
reference directory. Installation and public tests need none of those files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SNAPSHOT = "2026-10-05-nine-line-v1"
PREFIX = f"results/publication/{SNAPSHOT}/"
SNAPSHOT_REVISION = "1b748b220ec03aa9b6417ad3737f7542436672de"
LINES = tuple(f"line{i}" for i in range(1, 10))
MODELS = {
    "nandakumar_feh": "legacy_direct_metallicity_0p08",
    "nandakumar_no_feh": "legacy_direct_1d_0p08",
}
SCOPES = ("strict_strength_domain", "all_available")
COHORTS = ("exact_subset", "common_all9")
SOURCE_SHA256 = {
    PREFIX + "coefficients_all_methods.csv":
        "bfc0b011807f171e1bf8d14a47b559afd306d8ae27c205909f392e6c3cda330a",
    PREFIX + "subset_metrics.csv":
        "a340e1e79d92a0b0244a2162a3f683cef10726e25612efca52c5563e02cbfe30",
    PREFIX + "line_quality.csv":
        "21adaf4158078f815a1414b62e15d1a93051f0b033d25fa2373e360943b3ca83",
    PREFIX + "metadata.json":
        "51079148198763a987aae07bceb562ec1725eaceb4c53969da64970bbc66b508",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def export(reference_root: Path, output_dir: Path) -> dict[str, str]:
    """Verify the sealed four-file allowlist and write a new deterministic projection."""
    if Path(output_dir).resolve().is_relative_to(Path(reference_root).resolve()):
        raise ValueError("Export destination must be outside the read-only reference repository")
    source = {}
    for name, expected in SOURCE_SHA256.items():
        payload = (Path(reference_root) / name).read_bytes()
        if digest(payload) != expected:
            raise ValueError(f"Frozen HFS source SHA-256 mismatch: {name}")
        source[name] = payload
    metadata = json.loads(source[PREFIX + "metadata.json"])
    for name in ("coefficients_all_methods.csv", "subset_metrics.csv", "line_quality.csv"):
        if metadata["output_sha256"][name] != SOURCE_SHA256[PREFIX + name]:
            raise ValueError(f"Sealed snapshot manifest mismatch: {name}")

    def rows(name):
        return list(csv.DictReader(io.StringIO(source[PREFIX + name].decode("utf-8"))))

    coefficients = rows("coefficients_all_methods.csv")
    quality = rows("line_quality.csv")
    calibration = {
        "schema_version": 1,
        "calibration_id": SNAPSHOT,
        "measurement_source": "hfs",
        "temperature_scale_id": "nandakumar2023_igrins",
        "atomic_component_dataset": "hala2022_ab_generated",
        "declared_strength_selection": {
            "measurement_source": "gaussian",
            "minimum_angstrom": 0.08,
            "maximum_angstrom": None,
            "inclusive": True,
            "note": (
                "The frozen shared comparison selects using Gaussian-anchor strengths, "
                "never HFS areas. Target applicability is unknown without supplied "
                "Gaussian anchors."
            ),
        },
        "configurations": {},
    }
    for configuration, model_id in MODELS.items():
        selected = [r for r in coefficients if r["configuration"] == configuration
                    and r["measurement_source"] == "hfs"]
        if len(selected) != 9 or tuple(r["line_id"] for r in selected) != LINES:
            raise ValueError("Missing or duplicate frozen HFS coefficients")
        entry = {
            "model_id": model_id,
            "requires_feh": configuration == "nandakumar_feh",
            "fit_snapshot": selected[0]["fit_snapshot"],
            "validation_snapshot": selected[0]["loocv_snapshot"],
            "lines": {},
        }
        for row in selected:
            line_id = row["line_id"]
            if (row["model_id"] != model_id
                    or row["temperature_scale_id"] != calibration["temperature_scale_id"]
                    or row["fit_snapshot"] != entry["fit_snapshot"]
                    or row["loocv_snapshot"] != entry["validation_snapshot"]):
                raise ValueError("HFS coefficient identity mismatch")
            coverage = [r for r in quality if r["measurement_source"] == "hfs"
                        and r["model_id"] == model_id and r["line_id"] == line_id]
            if len(coverage) != 2 or {r["scope"] for r in coverage} != set(SCOPES):
                raise ValueError("Missing or duplicate HFS reference coverage")
            coverage_fields = (
                "reference_teff_min_k", "reference_teff_max_k",
                "reference_metallicity_min_dex", "reference_metallicity_max_dex",
            )
            for record in coverage:
                for field in ("a_k_per_angstrom", "b_k_per_dex", "c_k",
                              "strength_min_angstrom", "strength_max_angstrom"):
                    if record[field] != row[field]:
                        raise ValueError("HFS coefficient/coverage mismatch")
                if any(record[field] != coverage[0][field] for field in coverage_fields):
                    raise ValueError("Scope-dependent HFS coverage needs a new export schema")
            b = None if row["b_k_per_dex"] == "" else float(row["b_k_per_dex"])
            if (b is not None) != entry["requires_feh"]:
                raise ValueError("Unexpected HFS metallicity predictor")
            entry["lines"][line_id] = {
                "wavelength_air_angstrom": float(row["wavelength_air_angstrom"]),
                "n_training": int(row["n_training"]),
                "a_k_per_angstrom": float(row["a_k_per_angstrom"]),
                "b_k_per_dex": b,
                "c_k": float(row["c_k"]),
                "training_strength_range_angstrom": [float(row[field]) for field in (
                    "strength_min_angstrom", "strength_max_angstrom")],
                "reference_temperature_range_k": [float(coverage[0][field]) for field in (
                    "reference_teff_min_k", "reference_teff_max_k")],
                "reference_feh_range_dex": [float(coverage[0][field]) for field in (
                    "reference_metallicity_min_dex", "reference_metallicity_max_dex")],
            }
        calibration["configurations"][configuration] = entry

    subsets, identities = [], set()
    inverse = {model: name for name, model in MODELS.items()}
    for row in rows("subset_metrics.csv"):
        if row["measurement_source"] != "hfs" or row["model_id"] not in inverse:
            continue
        configuration = inverse[row["model_id"]]
        chosen = tuple(row["line_ids"].split("|"))
        if (not chosen or tuple(line for line in LINES if line in chosen) != chosen
                or len(chosen) != int(row["n_lines"])
                or row["temperature_scale_id"] != calibration["temperature_scale_id"]):
            raise ValueError("Invalid HFS subset identity")
        mask = sum(1 << LINES.index(line) for line in chosen)
        key = configuration, row["scope"], row["cohort"], mask
        if key in identities:
            raise ValueError("Duplicate HFS validation identity")
        identities.add(key)
        if int(row["n_stars"]) <= 0 or any(
            not math.isfinite(float(row[field])) for field in ("bias_k", "rmse_k")
        ):
            raise ValueError("Invalid frozen HFS validation statistics")
        subsets.append({
            "configuration": configuration, "scope": row["scope"], "cohort": row["cohort"],
            "subset_mask": mask, "n_stars": row["n_stars"],
            "bias_k": row["bias_k"], "rmse_k": row["rmse_k"],
        })
    expected = {(name, scope, cohort, mask) for name in MODELS for scope in SCOPES
                for cohort in COHORTS for mask in range(1, 512)}
    if identities != expected:
        raise ValueError("Frozen HFS validation must contain all 511 subsets in all eight groups")
    subsets.sort(key=lambda r: (
        tuple(MODELS).index(r["configuration"]), SCOPES.index(r["scope"]),
        COHORTS.index(r["cohort"]), r["subset_mask"],
    ))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(subsets[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(subsets)
    products = {
        "calibration.json": json_bytes(calibration),
        "subset_metrics.csv": buffer.getvalue().encode("utf-8"),
    }
    manifest = {
        "schema_version": 1,
        "calibration_id": SNAPSHOT,
        "measurement_source": "hfs",
        "source_repository": "https://github.com/thorsbro/scteff_paper",
        "snapshot_git_revision": SNAPSHOT_REVISION,
        "source_sha256": SOURCE_SHA256,
        "upstream_generator_code_sha256": metadata["code_sha256"],
        "upstream_software": metadata["software"],
        "export": {
            "models": MODELS, "coefficient_rows": 18, "validation_rows": len(subsets),
            "subset_encoding": "bit i-1 denotes linei; non-empty masks 1 through 511",
            "coefficient_conversion": "Original decimal strings to binary64; no refitting/rounding",
            "validation_conversion": "Original N, bias and RMSE strings retained verbatim",
            "selection_anchor": "Gaussian W >= 0.08 Angstrom, inclusive, no upper cut",
            "selection_definition_source": "Existing public frozen Gaussian provenance",
            "measurement_identity": (
                "Frozen model_id strings are shared across methods; measurement_source=hfs "
                "is an essential part of the calibration and validation identity."
            ),
            "excluded": (
                "Spectra, per-star products, catalogues, object lists, code and Git history"
            ),
        },
        "derived_sha256": {name: digest(payload) for name, payload in products.items()},
        "data_rights": (
            "No additional calibration-data licence or co-author permissions asserted; "
            "see DATA_NOTICE.md. Atomic data attribution is separate."
        ),
    }
    products["provenance.json"] = json_bytes(manifest)
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    for name, payload in products.items():
        (Path(output_dir) / name).write_bytes(payload)
    return {name: digest(payload) for name, payload in products.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.reference_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
