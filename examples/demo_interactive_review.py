"""Generate a complete, entirely analytical Sc I review demonstration.

No observations or observational templates are read. Profiles, noise, blends,
coverage truncation and masks are mathematical choices from make_review_demo.
Run with --scripted-review for an illustrative, deliberately explicit sequence
of manual decisions; this does not automatically review a user's real spectrum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from sc_teff_method import get_calibration
from sc_teff_method.synthetic import DEFAULT_SYNTHETIC_SEED, make_review_demo

LOADER_SETTINGS = {
    "wavelength_column": "wavelength", "flux_column": "flux",
    "uncertainty_column": "uncertainty", "mask_column": "exclude",
    "delimiter": ",", "header": True, "skiprows": 0,
}
CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}


def write_demo(output_dir: Path, *, seed: int = DEFAULT_SYNTHETIC_SEED,
               scripted_review: bool = False) -> Path:
    """Write a fresh synthetic input directory and optional reproducible review."""
    example = make_review_demo(seed=seed)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)
    spectrum_path = output_dir / "spectrum.csv"
    with spectrum_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("wavelength", "flux", "uncertainty", "exclude"))
        for wavelength, flux, uncertainty, mask in zip(
            example.wavelength_air_angstrom, example.flux, example.uncertainty,
            example.mask, strict=True,
        ):
            writer.writerow((format(float(wavelength), ".17g"), format(float(flux), ".17g"),
                             format(float(uncertainty), ".17g"), int(mask)))
    manifest = {
        "origin": "Entirely analytical Gaussian toy profiles; no observational inputs/templates.",
        "description": example.description,
        "seed": example.seed,
        "noise_sigma": 0.0015,
        "conventions": CONVENTIONS,
        "loader_settings": LOADER_SETTINGS,
        "sha256": hashlib.sha256(spectrum_path.read_bytes()).hexdigest(),
        "truth": {line: asdict(values) for line, values in example.truth.items()},
        "suggested_review": {
            "clean_lines": ["line1", "line5", "line7", "line9"],
            "pre_masked_line": "line2",
            "mask_blend_line": "line4",
            "reject_incomplete_line": "line3",
            "reject_bad_continuum_line": "line6",
            "reject_unedited_blend_line": "line8",
        },
        "limitations": (
            "This exercise demonstrates software behaviour, not atmospheric realism or "
            "temperature accuracy for an instrument. Manual selection may introduce bias."
        ),
    }
    if scripted_review:
        manifest["scripted_comparison"] = _scripted_review(spectrum_path, output_dir)
    (output_dir / "truth.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return spectrum_path


def _scripted_review(spectrum_path: Path, output_dir: Path) -> dict:
    from sc_teff_method.review import ReviewSession
    from sc_teff_method.spectrum import load_spectrum

    spectrum = load_spectrum(spectrum_path, **LOADER_SETTINGS, **CONVENTIONS)
    identity = {
        "filename": spectrum_path.name,
        "sha256": hashlib.sha256(spectrum_path.read_bytes()).hexdigest(),
        "loader_settings": LOADER_SETTINGS,
    }
    session = ReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    session.notes = (
        "Entirely analytical demonstration; scripted decisions illustrate manual review."
    )
    for line in ("line1", "line5", "line7", "line9"):
        session.decide(line, "accepted", note="Inspected the analytical Gaussian example.")
    session.decide("line2", "accepted",
                   note="Inspected the invented pre-masked wing; flags retained.")
    session.decide("line4", "accepted", note="Deliberate synthetic before-mask comparison only.")
    before_area = session.fits["line4"].area_angstrom
    before_temperature = session.reviewed.temperature.temperature_k
    rest = get_calibration().lines["line4"].wavelength_air_angstrom
    interval = (rest+0.15, rest+1.60)
    session.edit_line("line4", excluded_regions=[interval])
    invalidated = session.decisions["line4"].state == "pending"
    session.decide("line4", "accepted", note="Inspected after masking the invented neighbour.")
    for line, note in (
        ("line3", "Rejected the deliberately incomplete fitting-window coverage."),
        ("line6", "Rejected the imposed continuum error and poor residuals."),
        ("line8", "Rejected the unedited invented neighbouring absorption feature."),
    ):
        session.decide(line, "rejected", note=note)
    session.save(output_dir / "review.json", protected_paths=(spectrum_path,))
    return {
        "line4_mask_air_angstrom": list(interval),
        "line4_area_before_angstrom": before_area,
        "line4_area_after_angstrom": session.fits["line4"].area_angstrom,
        "acceptance_invalidated_after_mask": invalidated,
        "combined_temperature_before_k": before_temperature,
        "combined_temperature_after_k": session.reviewed.temperature.temperature_k,
        "comparison_subset": list(session.reviewed.included_line_ids),
        "note": "Same accepted subset before/after explicit reacceptance; numerical example only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--scripted-review", action="store_true")
    args = parser.parse_args()
    try:
        spectrum_path = write_demo(
            args.output_dir, seed=args.seed, scripted_review=args.scripted_review,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(f"Generated analytical synthetic review input: {spectrum_path}")
    print("No observational data were used. Inspect fits before accepting measurements.")


if __name__ == "__main__":
    main()
