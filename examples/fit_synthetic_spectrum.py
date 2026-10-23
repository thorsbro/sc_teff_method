"""Fit one entirely analytical spectrum and save local JSON, CSV and figures.

All profiles are made from mathematical Gaussians with invented parameters.
This example establishes numerical behaviour, not accuracy for stellar spectra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sc_teff_method import fit_spectrum, get_calibration, write_fit_outputs
from sc_teff_method.synthetic import SYNTHETIC_CASES, make_synthetic_spectrum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SYNTHETIC_CASES, default="clean")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-warned", action="store_true")
    parser.add_argument("--mask-illustrative-blend", action="store_true",
                        help="Mask the known invented neighbour in the blends case")
    args = parser.parse_args()
    if args.mask_illustrative_blend and args.case != "blends":
        parser.error("--mask-illustrative-blend applies only to --case blends")
    synthetic = make_synthetic_spectrum(args.case)
    regions = {}
    if args.mask_illustrative_blend:
        # This interval is selected from the invented toy blend's definition.
        centre = get_calibration().lines["line4"].wavelength_air_angstrom
        regions["line4"] = [(centre+0.45, centre+1.05)]
    result = fit_spectrum(
        synthetic.wavelength_air_angstrom, synthetic.flux,
        wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
        uncertainty=synthetic.uncertainty, mask=synthetic.mask, rv_kms=synthetic.rv_kms,
        feh=-0.2, excluded_regions=regions,
        quality_policy="include_warned" if args.include_warned else "exclude_warned",
    )
    try:
        write_fit_outputs(result, args.output_dir, plot_format="both",
                          spectrum_label=f"Analytical synthetic: {args.case}")
    except FileExistsError as exc:
        parser.exit(2, f"{exc}\n")
    for line, fit in result.fits.items():
        print(f"{line}: {fit.status}; area={fit.area_angstrom}; flags={fit.flags}")
    if result.temperature is not None:
        print(f"Teff = {result.temperature.temperature_k:.6f} K")
    else:
        print("No combined temperature under this inclusion policy.")
    print(f"Inspect {args.output_dir / 'diagnostics.png'} before interpreting the result.")


if __name__ == "__main__":
    main()
