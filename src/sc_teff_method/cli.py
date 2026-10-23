"""Command-line spectrum fitting; all outputs remain local unless the user shares them."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from ._data import CONFIGURATIONS, LINE_IDS, SCOPES


def _column(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _line_values(value: str, count: int) -> tuple[str, tuple[float, ...]]:
    parts = value.split(":")
    if len(parts) != count + 1 or parts[0] not in LINE_IDS:
        raise ValueError(f"Expected LINE_ID followed by {count} colon-separated numbers: {value}")
    return parts[0], tuple(float(part) for part in parts[1:])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sc-teff", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit a continuum-normalised air-Angstrom spectrum")
    fit.add_argument("spectrum", type=Path)
    fit.add_argument("--wavelength-unit", choices=["angstrom"], required=True)
    fit.add_argument("--wavelength-frame", choices=["air"], required=True)
    fit.add_argument("--continuum-normalised", action="store_true", required=True,
                     help="Confirm the input continuum has already been normalised to unity")
    fit.add_argument("--wavelength-column", type=_column, default=0)
    fit.add_argument("--flux-column", type=_column, default=1)
    fit.add_argument("--uncertainty-column", type=_column)
    fit.add_argument("--mask-column", type=_column, help="0=use, 1=exclude")
    fit.add_argument("--delimiter",
                     help="Single character; default is whitespace; use '\\t' for tab")
    fit.add_argument("--header", action="store_true", help="Read a header row (needed for names)")
    fit.add_argument("--skiprows", type=int, default=0)
    fit.add_argument("--rv-kms", type=float,
                     help="Positive=recession; apply relativistic rest correction")
    fit.add_argument("--configuration", choices=CONFIGURATIONS, default="nandakumar_feh")
    fit.add_argument("--feh", type=float,
                     help="Explicit [Fe/H] in dex; required for nandakumar_feh")
    fit.add_argument("--reference-temperature-k", type=float)
    fit.add_argument("--exclude-line", choices=LINE_IDS, action="append", default=[])
    fit.add_argument("--include-warned", action="store_true",
                     help="Explicitly include converged quality-flagged fits after inspection")
    fit.add_argument("--window", action="append", default=[], metavar="LINE:LOWER:UPPER",
                     help="Override a fitting window in corrected rest air Angstroms")
    fit.add_argument("--initial", action="append", default=[], metavar="LINE:A:CENTRE:SIGMA")
    fit.add_argument("--exclude-region", action="append", default=[], metavar="LINE:LOWER:UPPER")
    fit.add_argument("--fit-config", type=Path,
                     help="JSON: windows, initial_parameters, excluded_regions and/or options")
    fit.add_argument("--validation-scope", choices=SCOPES, default="strict_strength_domain")
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument("--overwrite", action="store_true",
                     help="Permit replacement of generated files in an existing output directory")
    fit.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    fit.add_argument("--label", default="Input spectrum",
                     help="Figure label; use a synthetic label for demos")
    return parser


def _fit_settings(args):
    from .fitting import FitOptions, GaussianParameters

    settings = {}
    if args.fit_config is not None:
        settings = json.loads(args.fit_config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("fit-config must contain a JSON object")
    unknown = set(settings) - {"windows", "initial_parameters", "excluded_regions", "options"}
    if unknown:
        raise ValueError(f"Unknown fit-config keys: {sorted(unknown)}")
    for key, value in settings.items():
        if not isinstance(value, dict):
            raise ValueError(f"fit-config {key} must be an object")
    windows = dict(settings.get("windows", {}))
    initial = {}
    for line, values in settings.get("initial_parameters", {}).items():
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("initial_parameters values must be [amplitude, centre, sigma]")
        initial[line] = GaussianParameters(*values)
    regions = {line: list(values) for line, values in settings.get("excluded_regions", {}).items()}
    for flag, target, size in ((args.window, windows, 2), (args.initial, initial, 3)):
        seen = set()
        for value in flag:
            line, numbers = _line_values(value, size)
            if line in seen:
                raise ValueError(f"Repeated window/initial override for {line}")
            seen.add(line)
            target[line] = GaussianParameters(*numbers) if size == 3 else numbers
    for value in args.exclude_region:
        line, numbers = _line_values(value, 2)
        regions.setdefault(line, []).append(numbers)
    options = FitOptions(**settings.get("options", {}))
    return windows, initial, regions, options


def _run_fit(args) -> int:
    from .outputs import write_fit_outputs
    from .spectrum import load_spectrum
    from .workflow import estimate_from_spectrum

    args.output_dir = args.output_dir.expanduser()
    args.spectrum = args.spectrum.expanduser()
    if args.fit_config is not None:
        args.fit_config = args.fit_config.expanduser()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}; use --overwrite"
        )
    input_paths = [args.spectrum] + ([args.fit_config] if args.fit_config is not None else [])
    output_names = ("summary.json", "lines.csv", "diagnostics.png", "diagnostics.pdf")
    for name in output_names:
        if any(path.resolve() == (args.output_dir / name).resolve() for path in input_paths):
            raise ValueError(
                "The output would replace an input spectrum or fit-config; choose another directory"
            )
    windows, initial, regions, options = _fit_settings(args)
    spectrum = load_spectrum(
        args.spectrum, wavelength_column=args.wavelength_column, flux_column=args.flux_column,
        uncertainty_column=args.uncertainty_column, mask_column=args.mask_column,
        delimiter="\t" if args.delimiter == "\\t" else args.delimiter,
        header=args.header, skiprows=args.skiprows,
        wavelength_unit=args.wavelength_unit, wavelength_frame=args.wavelength_frame,
        continuum_normalised=args.continuum_normalised, rv_kms=args.rv_kms,
    )
    result = estimate_from_spectrum(
        spectrum, configuration=args.configuration, feh=args.feh,
        reference_temperature_k=args.reference_temperature_k,
        windows=windows, initial_parameters=initial, excluded_regions=regions,
        excluded_lines=args.exclude_line, options=options,
        quality_policy="include_warned" if args.include_warned else "exclude_warned",
        validation_scope=args.validation_scope,
    )
    write_fit_outputs(
        result, args.output_dir, overwrite=args.overwrite, plot_format=args.plot_format,
        spectrum_label=args.label,
        input_metadata={
            "filename": args.spectrum.name,
            "sha256": hashlib.sha256(args.spectrum.read_bytes()).hexdigest(),
            "loader_settings": {
                "wavelength_column": args.wavelength_column,
                "flux_column": args.flux_column,
                "uncertainty_column": args.uncertainty_column,
                "mask_column": args.mask_column,
                "delimiter": "\t" if args.delimiter == "\\t" else args.delimiter,
                "header": args.header, "skiprows": args.skiprows,
            },
        },
    )
    if result.temperature is None:
        print("No combined temperature: no usable lines remain. Inspect the saved diagnostics.")
    else:
        print(f"Teff = {result.temperature.temperature_k:.6f} K "
              f"from {result.temperature.n_lines} lines ({result.configuration})")
        print(f"Exact-subset empirical validation: N={result.performance.n_stars}, "
              f"bias={result.performance.bias_k:.6f} K, RMSE={result.performance.rmse_k:.6f} K")
        print(result.temperature.uncertainty_note)
    for warning in result.warnings:
        print(f"Warning: {warning}")
    for warning in result.calibration_warnings:
        print(f"Warning [{warning.code}]: {warning.message}")
    print(f"Outputs: {args.output_dir}")
    return 0 if result.temperature is not None else 3


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run_fit(args)
    except (ValueError, TypeError, OSError, ImportError) as exc:
        print(f"sc-teff: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
