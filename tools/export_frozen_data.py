"""Export an explicit allowlist of frozen numerical products, without refitting.

Maintainer tool only; the installed package and public tests never need the
private repository. Reads only the paths below and writes a new output directory.
No spectra, per-star measurements, identifiers, catalogues or source code are
exported. Source code and documentation are hashed for provenance only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tomllib
from pathlib import Path

SNAPSHOT = "2026-10-05-nine-line-v1"
PREFIX = f"results/publication/{SNAPSHOT}/"
SNAPSHOT_REVISION = "1b748b220ec03aa9b6417ad3737f7542436672de"
REFERENCE_REVISION = "ad70e2e8c9c6ac458168e862d609233e2c564af3"
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
    "docs/nine_line_quality.md":
        "7b13302be7dc4388560bbba0b5c2e72c0f63df5df42004ce235af41ed83238b3",
    "src/scteff/line_quality.py":
        "0ddc1863e05c43defd24f10583184696b005a658354a9cef2e3106dd94f75d81",
    "tests/test_line_quality.py":
        "8a52298632457e8bf9cb07fa23f11470e658d91a4a707f10060716ac01cc26a2",
    "config/lines.csv":
        "fbe152f639500d5e48f37ae47379c88bfaeca4179953a3b5f372c6d4a2f69162",
    "config/calibration_models.toml":
        "11d27fe694807f18fbab575e67abb33d194efe4cf3d3018318b02c15dfe352e0",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def export(reference_root: Path, output_dir: Path) -> dict[str, str]:
    """Verify frozen inputs, then project only the supported public numbers."""
    source = {}
    for name, expected in SOURCE_SHA256.items():
        payload = (reference_root / name).read_bytes()
        if digest(payload) != expected:
            raise ValueError(f"Reference source SHA-256 mismatch: {name}")
        source[name] = payload
    metadata = json.loads(source[PREFIX + "metadata.json"])
    for name in ("coefficients_all_methods.csv", "subset_metrics.csv", "line_quality.csv"):
        if metadata["output_sha256"][name] != SOURCE_SHA256[PREFIX + name]:
            raise ValueError(f"Upstream sealed manifest mismatch: {name}")

    def rows(name: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(source[name].decode("utf-8"))))

    wavelengths = {r["line_id"]: float(r["wavelength_air_angstrom"])
                   for r in rows("config/lines.csv")}
    specifications = {
        r["model_id"]: r
        for r in tomllib.loads(source["config/calibration_models.toml"].decode())["models"]
    }
    coefficients = rows(PREFIX + "coefficients_all_methods.csv")
    quality = rows(PREFIX + "line_quality.csv")
    calibration = {
        "schema_version": 1,
        "calibration_id": SNAPSHOT,
        "measurement_source": "gaussian",
        "temperature_scale_id": "nandakumar2023_igrins",
        "declared_strength_selection": {
            "minimum_angstrom": 0.08, "maximum_angstrom": None, "inclusive": True,
        },
        "configurations": {},
    }
    for configuration, model_id in MODELS.items():
        specification = specifications[model_id]
        if (specification["strength_min"] != 0.08 or specification["strength_max"] != "none"
                or specification["temperature_min"] != "none"
                or specification["temperature_max"] != "none"):
            raise ValueError("Unexpected declared model-selection cut")
        selected = [r for r in coefficients if r["configuration"] == configuration
                    and r["measurement_source"] == "gaussian"]
        if len(selected) != 9 or tuple(r["line_id"] for r in selected) != LINES:
            raise ValueError("Incomplete or duplicate frozen Gaussian coefficients")
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
                    or float(row["wavelength_air_angstrom"]) != wavelengths[line_id]):
                raise ValueError("Coefficient identity, scale or wavelength mismatch")
            coverage = [r for r in quality if r["measurement_source"] == "gaussian"
                        and r["model_id"] == model_id and r["line_id"] == line_id]
            if len(coverage) != 2 or {r["scope"] for r in coverage} != set(SCOPES):
                raise ValueError("Missing or duplicate reference coverage")
            coverage_fields = (
                "reference_teff_min_k", "reference_teff_max_k",
                "reference_metallicity_min_dex", "reference_metallicity_max_dex",
            )
            for r in coverage:
                for field in ("a_k_per_angstrom", "b_k_per_dex", "c_k",
                              "strength_min_angstrom", "strength_max_angstrom"):
                    if r[field] != row[field]:
                        raise ValueError("Coefficient/quality mismatch")
                if any(r[k] != coverage[0][k] for k in coverage_fields):
                    raise ValueError("Scope-dependent coverage requires a new export schema")
            b = None if row["b_k_per_dex"] == "" else float(row["b_k_per_dex"])
            if (b is not None) != entry["requires_feh"]:
                raise ValueError("Unexpected metallicity predictor")
            entry["lines"][line_id] = {
                "wavelength_air_angstrom": float(row["wavelength_air_angstrom"]),
                "n_training": int(row["n_training"]),
                "a_k_per_angstrom": float(row["a_k_per_angstrom"]),
                "b_k_per_dex": b,
                "c_k": float(row["c_k"]),
                "training_strength_range_angstrom": [float(row[k]) for k in (
                    "strength_min_angstrom", "strength_max_angstrom")],
                "reference_temperature_range_k": [float(coverage[0][k]) for k in (
                    "reference_teff_min_k", "reference_teff_max_k")],
                "reference_feh_range_dex": [float(coverage[0][k]) for k in (
                    "reference_metallicity_min_dex", "reference_metallicity_max_dex")],
            }
        calibration["configurations"][configuration] = entry

    subsets = []
    inverse = {v: k for k, v in MODELS.items()}
    identities = set()
    for row in rows(PREFIX + "subset_metrics.csv"):
        if row["measurement_source"] != "gaussian" or row["model_id"] not in inverse:
            continue
        configuration = inverse[row["model_id"]]
        chosen = tuple(row["line_ids"].split("|"))
        if (not chosen or tuple(line for line in LINES if line in chosen) != chosen
                or len(chosen) != int(row["n_lines"])
                or row["temperature_scale_id"] != calibration["temperature_scale_id"]):
            raise ValueError("Invalid fixed-subset identity or scale")
        mask = sum(1 << LINES.index(line) for line in chosen)
        identity = (configuration, row["scope"], row["cohort"], mask)
        if identity in identities:
            raise ValueError("Duplicate subset identity")
        identities.add(identity)
        if int(row["n_stars"]) <= 0 or any(
            not math.isfinite(float(row[k])) for k in ("bias_k", "rmse_k")
        ):
            raise ValueError("Invalid held-out statistics")
        subsets.append({
            "configuration": configuration,
            "scope": row["scope"],
            "cohort": row["cohort"],
            "subset_mask": mask,
            "n_stars": row["n_stars"],
            "bias_k": row["bias_k"],
            "rmse_k": row["rmse_k"],
        })
    expected = {(configuration, scope, cohort, mask) for configuration in MODELS
                for scope in SCOPES for cohort in COHORTS for mask in range(1, 512)}
    if identities != expected:
        raise ValueError("Not all 511 subsets exist for every supported model/scope/cohort")
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
        "source_repository": "https://github.com/thorsbro/scteff_paper",
        "source_visibility": "private",
        "snapshot_git_revision": SNAPSHOT_REVISION,
        "reference_git_revision": REFERENCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "upstream_generator_code_sha256": metadata["code_sha256"],
        "upstream_software": metadata["software"],
        "export": {
            "measurement_source": "gaussian",
            "models": MODELS,
            "coefficient_rows": 18,
            "validation_rows": len(subsets),
            "subset_encoding": "bit i-1 denotes linei, masks 1 through 511",
            "coefficient_conversion": (
                "Decimal source strings to binary64; no refitting or rounding"
            ),
            "validation_conversion": "Original N, bias and RMSE strings retained verbatim",
            "coverage": "Training strengths; retained held-out temperature/metallicity references",
            "excluded": "Spectra, per-star products, catalogues, source code and Git history",
        },
        "derived_sha256": {name: digest(payload) for name, payload in products.items()},
        "data_rights": (
            "No additional data license or co-author permissions asserted; see DATA_NOTICE.md"
        ),
    }
    products["provenance.json"] = json_bytes(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in products.items():
        (output_dir / name).write_bytes(payload)
    return {name: digest(payload) for name, payload in products.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory")
    args = parser.parse_args()
    hashes = export(args.reference_root, args.output_dir)
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
