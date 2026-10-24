"""Write a deterministic analytical HFS spectrum; no observed input is used."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from sc_teff_method.hfs_synthetic import HFS_SYNTHETIC_CASES, make_hfs_synthetic_spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=HFS_SYNTHETIC_CASES, default="clean")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    example = make_hfs_synthetic_spectrum(args.case, seed=args.seed)
    destination = args.output_dir.expanduser()
    destination.mkdir(parents=True, exist_ok=False)
    source = destination/"spectrum.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        header = ["wavelength", "flux", "exclude"]
        if example.uncertainty is not None:
            header.append("uncertainty")
        writer.writerow(header)
        for index, (wavelength, flux, mask) in enumerate(zip(
            example.wavelength_air_angstrom, example.flux, example.mask, strict=True,
        )):
            row = [format(float(wavelength), ".17g"), format(float(flux), ".17g"), int(mask)]
            if example.uncertainty is not None:
                row.append(format(float(example.uncertainty[index]), ".17g"))
            writer.writerow(row)
    manifest = {
        "origin": (
            "Entirely analytical HFS mixtures; no observed spectra or observational templates."
        ),
        "case": example.name, "seed": example.seed, "description": example.description,
        "measurement_method": "hfs", "atomic_dataset_id": example.atomic_dataset_id,
        "wavelength_unit": "angstrom", "wavelength_frame": "air",
        "continuum_normalised": True, "rv_kms": example.rv_kms,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "truth": {line: asdict(parameters) for line, parameters in example.truth.items()},
        "limitations": (
            "Artificial parameters, Gaussian broadening and optional seeded noise do not "
            "represent a cool-giant atmosphere or prove instrument-transfer accuracy. "
            "Full HFS areas use sqrt(2*pi)*A*sigma, not a Gaussian-area calibration."
        ),
    }
    destination.joinpath("truth.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8",
    )
    print(f"Generated analytical HFS input: {source}")
    print("Use HFS-specific fitting/calibration. Inspect all fits before interpreting Teff.")
    if example.rv_kms is not None:
        print(f"The invented raw-grid redshift requires explicit --rv-kms {example.rv_kms:g}.")


if __name__ == "__main__":
    main()
