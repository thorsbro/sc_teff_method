"""Generate an entirely analytical local-continuum review exercise.

All Gaussian profiles, constants, slopes, blends, masks and optional noise are
invented mathematical choices. No observation or observational template is read.
The optional scripted decisions illustrate the interface's manual workflow;
they are not a continuum-selection algorithm for a user's spectrum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from sc_teff_method.synthetic import make_continuum_demo

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}


def write_demo(
    output_dir: Path, *, seed: int = 20261024, noise_sigma: float = 0.0,
    scripted_review: bool = False,
) -> Path:
    """Write fresh toy input/truth; never overwrite an existing directory."""
    example = make_continuum_demo(seed=seed, noise_sigma=noise_sigma)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)
    spectrum_path = output_dir / "spectrum.csv"
    loader = {
        "wavelength_column": "wavelength", "flux_column": "flux", "mask_column": "exclude",
        "uncertainty_column": "uncertainty" if example.uncertainty is not None else None,
        "delimiter": ",", "header": True, "skiprows": 0,
    }
    with spectrum_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        header = ["wavelength", "flux", "exclude"]
        if example.uncertainty is not None:
            header.append("uncertainty")
        writer.writerow(header)
        for i, (wavelength, flux, mask) in enumerate(zip(
            example.wavelength_air_angstrom, example.flux, example.mask, strict=True,
        )):
            values = [format(float(wavelength), ".17g"), format(float(flux), ".17g"), int(mask)]
            if example.uncertainty is not None:
                values.append(format(float(example.uncertainty[i]), ".17g"))
            writer.writerow(values)
    manifest = {
        "origin": "Entirely analytical Gaussian toy profiles; no observational inputs/templates.",
        "description": example.description,
        "seed": example.seed,
        "noise_sigma_in_corrected_flux": noise_sigma,
        "conventions": CONVENTIONS,
        "loader_settings": loader,
        "sha256": hashlib.sha256(spectrum_path.read_bytes()).hexdigest(),
        "truth": {line: asdict(value) for line, value in example.truth.items()},
        "continuum_truth": {
            line: asdict(value) for line, value in example.continuum_truth.items()
        },
        "incorrect_line5_example_anchors": example.incorrect_anchor_example,
        "suggested_review": {
            "unity_line": "line1", "constant_line": "line2", "sloped_line": "line3",
            "blended_line": "line4", "incorrect_anchor_exercise": "line5",
            "masked_line": "line7", "incomplete_coverage": "line8",
        },
        "limitations": (
            "These mathematically known continua are teaching inputs, not inferred estimates. "
            "A changed area or temperature is not necessarily better. Formal fit errors omit "
            "continuum placement uncertainty; calibration RMSE does not validate new continuum "
            "choices or post-hoc selection. Synthetic tests do not establish instrument accuracy."
        ),
    }
    if scripted_review:
        manifest["scripted_comparison"] = _scripted_review(
            spectrum_path, output_dir, example, loader,
        )
    (output_dir / "truth.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return spectrum_path


def _scripted_review(spectrum_path, output_dir, example, loader) -> dict:
    from sc_teff_method.continuum import ContinuumCorrection
    from sc_teff_method.review import ReviewSession
    from sc_teff_method.review_io import load_spectrum_identity

    spectrum, identity = load_spectrum_identity(
        spectrum_path, loader_settings=loader, **CONVENTIONS,
    )
    session = ReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    session.notes = (
        "Analytical continuum demonstration only. The prescribed mathematical truth is known; "
        "scripted acceptance illustrates explicit decisions, not automatic scientific review."
    )
    original = {line: fit.area_angstrom for line, fit in session.fits.items()}
    for line in ("line1", "line2", "line3", "line5", "line6", "line7", "line9"):
        truth = example.continuum_truth[line]
        if truth.mode != "none":
            session.edit_line(
                line, continuum=ContinuumCorrection(enabled=True, **asdict(truth)),
            )
        session.decide(
            line, "accepted", note="Explicitly inspected the prescribed analytical continuum.",
        )
    session.decide("line4", "rejected", note="Invented blend left unmasked in this exercise.")
    session.decide("line8", "rejected", note="Deliberately incomplete lower continuum coverage.")
    correct_temperature = session.reviewed.temperature.temperature_k
    correct_line5 = session.fits["line5"].area_angstrom
    session.edit_line("line5", continuum=ContinuumCorrection(
        enabled=True, mode="linear", anchors=example.incorrect_anchor_example,
    ))
    invalidated = session.decisions["line5"].state == "pending"
    session.decide("line5", "accepted", note=(
        "Deliberate incorrect-anchor comparison using a known mathematical truth; "
        "this changed answer is not claimed to be scientifically better."
    ))
    incorrect_area = session.fits["line5"].area_angstrom
    incorrect_temperature = session.reviewed.temperature.temperature_k
    session.reset_continuum("line5")
    session.decide("line5", "accepted", note="Restored the known analytical unity continuum.")
    session.save(output_dir / "review.json", protected_paths=(spectrum_path,))
    return {
        "areas_before_correction_angstrom": original,
        "areas_corrected_angstrom": {
            line: fit.area_angstrom for line, fit in session.fits.items()
        },
        "line5_correct_area_angstrom": correct_line5,
        "line5_incorrect_anchor_area_angstrom": incorrect_area,
        "acceptance_invalidated_after_continuum_edit": invalidated,
        "correct_continuum_temperature_k": correct_temperature,
        "incorrect_anchor_temperature_k": incorrect_temperature,
        "comparison_subset": list(session.reviewed.included_line_ids),
        "saved_review": "Prescribed correct continua; line5 restored to unity after comparison.",
        "note": (
            "Same accepted subset after explicit reapproval; a changed answer is not validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--scripted-review", action="store_true")
    args = parser.parse_args()
    try:
        path = write_demo(
            args.output_dir, seed=args.seed, noise_sigma=args.noise_sigma,
            scripted_review=args.scripted_review,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(f"Generated entirely analytical continuum-review input: {path}")
    print("No observational data used. Continuum choices require scientific inspection.")


if __name__ == "__main__":
    main()
