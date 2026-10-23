"""Generate analytical toy CSV spectra and their exact Gaussian truth.

No observed spectrum, extract, template or individual-star measurement is read.
All amplitudes, widths, sampling and contaminants are invented. These examples
verify numerical behaviour and do not validate the method on real stars.

After installing sc-teff-method[spectra], run:
    python examples/generate_synthetic_spectra.py --output-dir synthetic_examples
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from sc_teff_method.synthetic import (
    DEFAULT_SYNTHETIC_SEED,
    SYNTHETIC_CASES,
    make_synthetic_spectrum,
)


def write_examples(output_dir: Path, *, case: str = "all", seed: int = DEFAULT_SYNTHETIC_SEED):
    """Write new files, refusing an output directory that already exists."""
    if case != "all" and case not in SYNTHETIC_CASES:
        raise ValueError(f"case must be 'all' or one of {SYNTHETIC_CASES}")
    names = SYNTHETIC_CASES if case == "all" else (case,)
    scenarios = [make_synthetic_spectrum(name, seed=seed) for name in names]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "origin": "Entirely analytical synthetic spectra; no observational inputs or templates.",
        "limitations": (
            "Independent Gaussian toy profiles do not reproduce cool-giant atmospheres or "
            "establish accuracy for real spectra or arbitrary instruments."
        ),
        "wavelength_unit": "angstrom",
        "wavelength_frame": "air",
        "mask_convention": "exclude=1 means exclude the pixel; exclude=0 means retain it",
        "rv_convention": (
            "Positive km/s means recession. Rest wavelengths equal input wavelengths divided "
            "by sqrt((1+v/c)/(1-v/c)), with c=299792.458 km/s."
        ),
        "cases": {},
    }
    for scenario in scenarios:
        fields = ["wavelength", "flux"]
        if scenario.uncertainty is not None:
            fields.append("uncertainty")
        fields.append("exclude")
        with output_dir.joinpath(f"{scenario.name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fields)
            for i, wavelength in enumerate(scenario.wavelength_air_angstrom):
                row = [format(float(wavelength), ".17g"), format(float(scenario.flux[i]), ".17g")]
                if scenario.uncertainty is not None:
                    row.append(format(float(scenario.uncertainty[i]), ".17g"))
                row.append(int(scenario.mask[i]))
                writer.writerow(row)
        manifest["cases"][scenario.name] = {
            "description": scenario.description,
            "seed": scenario.seed,
            "csv_file": f"{scenario.name}.csv",
            "columns": fields,
            "rv_kms": scenario.rv_kms,
            "truth_frame": "rest air angstrom",
            "truth": {line_id: asdict(line) for line_id, line in scenario.truth.items()},
        }
    output_dir.joinpath("truth.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return tuple(output_dir.joinpath(f"{name}.csv") for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("all", *SYNTHETIC_CASES), default="all")
    parser.add_argument("--seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    args = parser.parse_args()
    try:
        written = write_examples(args.output_dir, case=args.case, seed=args.seed)
    except (FileExistsError, ValueError, TypeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(
        f"Wrote {len(written)} entirely analytical toy spectra and truth.json to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
