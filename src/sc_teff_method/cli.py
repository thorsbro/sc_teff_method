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


def _add_spectrum_arguments(fit, *, declarations: bool = True, quality_policy: bool = True):
    fit.add_argument("spectrum", type=Path)
    fit.add_argument("--wavelength-unit", choices=["angstrom"], required=declarations)
    fit.add_argument("--wavelength-frame", choices=["air"], required=declarations)
    fit.add_argument(
        "--continuum-normalised",
        action="store_true",
        required=declarations,
        default=False if declarations else None,
        help="Confirm the input continuum has already been normalised to unity",
    )
    fit.add_argument("--wavelength-column", type=_column, default=0 if declarations else None)
    fit.add_argument("--flux-column", type=_column, default=1 if declarations else None)
    fit.add_argument("--uncertainty-column", type=_column)
    fit.add_argument("--mask-column", type=_column, help="0=use, 1=exclude")
    fit.add_argument(
        "--delimiter", help="Single character; default is whitespace; use '\\t' for tab"
    )
    fit.add_argument(
        "--header",
        action="store_true",
        default=False if declarations else None,
        help="Read a header row (needed for names)",
    )
    fit.add_argument("--skiprows", type=int, default=0 if declarations else None)
    fit.add_argument(
        "--rv-kms", type=float, help="Positive=recession; apply relativistic rest correction"
    )
    fit.add_argument(
        "--configuration",
        choices=CONFIGURATIONS,
        default="nandakumar_feh" if declarations else None,
    )
    fit.add_argument(
        "--feh", type=float, help="Explicit [Fe/H] in dex; required for nandakumar_feh"
    )
    fit.add_argument("--reference-temperature-k", type=float)
    fit.add_argument("--exclude-line", choices=LINE_IDS, action="append", default=[])
    if quality_policy:
        fit.add_argument(
            "--include-warned",
            action="store_true",
            help="Explicitly include converged quality-flagged fits after inspection",
        )
    fit.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="LINE:LOWER:UPPER",
        help="Override a fitting window in corrected rest air Angstroms",
    )
    fit.add_argument("--initial", action="append", default=[], metavar="LINE:A:CENTRE:SIGMA")
    fit.add_argument("--exclude-region", action="append", default=[], metavar="LINE:LOWER:UPPER")
    fit.add_argument(
        "--fit-config",
        type=Path,
        help="JSON: windows, initial_parameters, excluded_regions and/or options",
    )
    fit.add_argument(
        "--validation-scope",
        choices=SCOPES,
        default="strict_strength_domain" if declarations else None,
    )


def _add_output_arguments(fit):
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of generated files in an existing output directory",
    )
    fit.add_argument("--plot-format", choices=["png", "pdf", "both"], default="png")
    fit.add_argument(
        "--label", default="Input spectrum", help="Figure label; use a synthetic label for demos"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sc-teff", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit a continuum-normalised air-Angstrom spectrum")
    _add_spectrum_arguments(fit)
    _add_output_arguments(fit)
    hfs = commands.add_parser("fit-hfs", help="Fit fixed A+B HFS profiles and HFS temperatures")
    _add_spectrum_arguments(hfs)
    _add_output_arguments(hfs)
    hfs.add_argument(
        "--gaussian-anchor-areas", type=Path,
        help="JSON of separately measured Gaussian areas for strict-domain applicability",
    )
    hfs.add_argument(
        "--continuum-config", type=Path,
        help="JSON mapping line IDs to PR 4 manual continuum controls",
    )
    hfs.add_argument("--continuum-warning-deviation", type=float, default=0.1)
    hfs.add_argument("--no-continuum-extrapolation-warning", action="store_true")
    inspect = commands.add_parser(
        "inspect",
        help="Inspect and manually review fits in a local browser",
    )
    _add_spectrum_arguments(inspect, declarations=False, quality_policy=False)
    inspect.add_argument(
        "--review-file", type=Path, help="Restore controls and decisions for this input"
    )
    inspect.add_argument(
        "--save-review", type=Path, help="Initial Save review path in the interface"
    )
    inspect.add_argument("--output-dir", type=Path, default=Path("review-results"))
    inspect.add_argument(
        "--port", type=int, default=0, help="Local port; default chooses a free port"
    )
    inspect.add_argument(
        "--no-browser", action="store_true", help="Print local URL without opening a tab"
    )
    inspect.add_argument("--label", default="Input spectrum")
    inspect.add_argument(
        "--continuum-warning-deviation", type=float,
        help="Provisional warning threshold for abs(C-1); default 0.1",
    )
    inspect.add_argument(
        "--no-continuum-extrapolation-warning", action="store_true", default=None,
        help="Disable the provisional warning for linear anchors inside the fit window",
    )
    replay = commands.add_parser("replay", help="Reproduce a saved review without a browser")
    replay.add_argument("spectrum", type=Path)
    replay.add_argument("--review-file", type=Path, required=True)
    _add_output_arguments(replay)
    return parser


def _fit_settings(args, *, options_class=None):
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
    factory = FitOptions if options_class is None else options_class
    options = factory(**settings.get("options", {}))
    return windows, initial, regions, options


def _run_fit(args) -> int:
    from .outputs import write_fit_outputs
    from .review_io import load_spectrum_identity, protect_output_inputs
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
    protect_output_inputs(args.output_dir, input_paths, include_review=False)
    windows, initial, regions, options = _fit_settings(args)
    spectrum, identity = load_spectrum_identity(
        args.spectrum,
        loader_settings=_loader_settings(args),
        wavelength_unit=args.wavelength_unit,
        wavelength_frame=args.wavelength_frame,
        continuum_normalised=args.continuum_normalised,
        rv_kms=args.rv_kms,
    )
    result = estimate_from_spectrum(
        spectrum,
        configuration=args.configuration,
        feh=args.feh,
        reference_temperature_k=args.reference_temperature_k,
        windows=windows,
        initial_parameters=initial,
        excluded_regions=regions,
        excluded_lines=args.exclude_line,
        options=options,
        quality_policy="include_warned" if args.include_warned else "exclude_warned",
        validation_scope=args.validation_scope,
    )
    write_fit_outputs(
        result,
        args.output_dir,
        overwrite=args.overwrite,
        plot_format=args.plot_format,
        spectrum_label=args.label,
        input_metadata=identity,
    )
    if result.temperature is None:
        print("No combined temperature: no usable lines remain. Inspect the saved diagnostics.")
    else:
        print(
            f"Teff = {result.temperature.temperature_k:.6f} K "
            f"from {result.temperature.n_lines} lines ({result.configuration})"
        )
        print(
            f"Exact-subset empirical validation: N={result.performance.n_stars}, "
            f"bias={result.performance.bias_k:.6f} K, RMSE={result.performance.rmse_k:.6f} K"
        )
        print(result.temperature.uncertainty_note)
    for warning in result.warnings:
        print(f"Warning: {warning}")
    for warning in result.calibration_warnings:
        print(f"Warning [{warning.code}]: {warning.message}")
    print(f"Outputs: {args.output_dir}")
    return 0 if result.temperature is not None else 3


def _run_hfs(args) -> int:
    from .continuum import ContinuumCorrection, ContinuumOptions
    from .hfs_fitting import HFSFitOptions
    from .hfs_outputs import write_hfs_outputs
    from .hfs_workflow import estimate_hfs_from_spectrum
    from .review_io import load_spectrum_identity, protect_output_inputs

    args.spectrum = args.spectrum.expanduser()
    args.output_dir = args.output_dir.expanduser()
    protected = [args.spectrum]
    for name in ("fit_config", "gaussian_anchor_areas", "continuum_config"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser()
            setattr(args, name, path)
            protected.append(path)
    protect_output_inputs(args.output_dir, protected, include_review=False)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}; use --overwrite"
        )
    windows, initial, regions, options = _fit_settings(args, options_class=HFSFitOptions)
    anchors = (None if args.gaussian_anchor_areas is None else
               json.loads(args.gaussian_anchor_areas.read_text(encoding="utf-8")))
    corrections = {}
    if args.continuum_config is not None:
        controls = json.loads(args.continuum_config.read_text(encoding="utf-8"))
        if not isinstance(controls, dict) or any(line not in LINE_IDS for line in controls):
            raise ValueError("continuum-config must map line1 through line9 to manual controls")
        for line, values in controls.items():
            if not isinstance(values, dict):
                raise ValueError("Each continuum-config entry must be an object")
            corrections[line] = ContinuumCorrection(**values)
    spectrum, identity = load_spectrum_identity(
        args.spectrum, loader_settings=_loader_settings(args), rv_kms=args.rv_kms,
        wavelength_unit=args.wavelength_unit, wavelength_frame=args.wavelength_frame,
        continuum_normalised=args.continuum_normalised,
    )
    result = estimate_hfs_from_spectrum(
        spectrum, configuration=args.configuration, feh=args.feh,
        reference_temperature_k=args.reference_temperature_k,
        windows=windows, initial_parameters=initial, excluded_regions=regions,
        excluded_lines=args.exclude_line, options=options,
        quality_policy="include_warned" if args.include_warned else "exclude_warned",
        validation_scope=args.validation_scope, gaussian_anchor_areas=anchors,
        continuum_corrections=corrections, continuum_options=ContinuumOptions(
            max_deviation_warning=args.continuum_warning_deviation,
            warn_extrapolation=not args.no_continuum_extrapolation_warning,
        ),
    )
    write_hfs_outputs(
        result, args.output_dir, overwrite=args.overwrite, plot_format=args.plot_format,
        spectrum_label=args.label, input_metadata=identity, protected_paths=protected,
    )
    if result.temperature is None:
        print("No combined HFS temperature: no usable lines remain; inspect saved diagnostics.")
    else:
        print(f"HFS Teff = {result.temperature.temperature_k:.6f} K "
              f"from {result.temperature.n_lines} lines ({result.configuration})")
        print("Reference procedure: unweighted HFS mixture; fixed laboratory A+B pattern.")
        print(f"Frozen exact-subset {result.validation_scope} statistics: "
              f"N={result.performance.n_stars}, bias={result.performance.bias_k:.6f} K, "
              f"RMSE={result.performance.rmse_k:.6f} K")
        print(f"Strict-domain Gaussian-anchor applicability: "
              f"{result.strict_strength_domain_applicable}")
        print(result.temperature.uncertainty_note)
    for warning in result.warnings:
        print(f"Warning: {warning}")
    for warning in result.calibration_warnings:
        print(f"Warning [{warning.code}]: {warning.message}")
    print(f"Outputs: {args.output_dir}")
    return 0 if result.temperature is not None else 3


def _loader_settings(args) -> dict:
    from .review_io import LOADER_DEFAULTS

    settings = {
        name: (getattr(args, name) if getattr(args, name) is not None else default)
        for name, default in LOADER_DEFAULTS.items()
    }
    if settings["delimiter"] == "\\t":
        settings["delimiter"] = "\t"
    return settings


def _restore_review(spectrum_path, review_path):
    from .review import ReviewSession, read_review
    from .review_io import load_spectrum_identity

    payload = read_review(review_path)
    saved = payload["input"]
    source = Path(spectrum_path).expanduser()
    if hashlib.sha256(source.read_bytes()).hexdigest() != saved["sha256"]:
        raise ValueError("Input spectrum SHA-256 checksum does not match the saved review")
    spectrum, identity = load_spectrum_identity(
        source,
        loader_settings=saved["loader_settings"],
        rv_kms=saved["rv_kms"],
        wavelength_unit=saved["wavelength_unit"],
        wavelength_frame=saved["wavelength_frame"],
        continuum_normalised=saved["continuum_normalised"],
    )
    return ReviewSession.from_dict(spectrum, payload, input_identity=identity)


def _run_inspect(args) -> int:
    from .continuum import ContinuumOptions
    from .inspection import serve_inspection
    from .review import ReviewSession
    from .review_io import load_spectrum_identity

    args.spectrum = args.spectrum.expanduser()
    if args.fit_config is not None:
        args.fit_config = args.fit_config.expanduser()
    protected = [args.spectrum] + ([args.fit_config] if args.fit_config is not None else [])
    if args.review_file is not None:
        scientific_arguments = (
            "wavelength_unit",
            "wavelength_frame",
            "continuum_normalised",
            "wavelength_column",
            "flux_column",
            "uncertainty_column",
            "mask_column",
            "delimiter",
            "header",
            "skiprows",
            "rv_kms",
            "configuration",
            "feh",
            "reference_temperature_k",
            "fit_config",
            "validation_scope",
            "continuum_warning_deviation",
            "no_continuum_extrapolation_warning",
        )
        if (
            any(getattr(args, name) is not None for name in scientific_arguments)
            or args.window
            or args.initial
            or args.exclude_region
            or args.exclude_line
        ):
            raise ValueError(
                "A restored review uses its saved controls; omit fitting/input overrides"
            )
        session = _restore_review(args.spectrum, args.review_file.expanduser())
    else:
        if (
            args.wavelength_unit != "angstrom"
            or args.wavelength_frame != "air"
            or args.continuum_normalised is not True
        ):
            raise ValueError(
                "A new review requires --wavelength-unit angstrom --wavelength-frame air "
                "--continuum-normalised"
            )
        windows, initial, regions, options = _fit_settings(args)
        spectrum, identity = load_spectrum_identity(
            args.spectrum,
            loader_settings=_loader_settings(args),
            rv_kms=args.rv_kms,
            wavelength_unit=args.wavelength_unit,
            wavelength_frame=args.wavelength_frame,
            continuum_normalised=args.continuum_normalised,
        )
        session = ReviewSession.create(
            spectrum,
            input_identity=identity,
            configuration=args.configuration or "nandakumar_feh",
            feh=args.feh,
            reference_temperature_k=args.reference_temperature_k,
            validation_scope=args.validation_scope or "strict_strength_domain",
            windows=windows,
            initial_parameters=initial,
            excluded_regions=regions,
            excluded_lines=args.exclude_line,
            options=options,
            continuum_options=ContinuumOptions(
                max_deviation_warning=(
                    0.1 if args.continuum_warning_deviation is None
                    else args.continuum_warning_deviation
                ),
                warn_extrapolation=not args.no_continuum_extrapolation_warning,
            ),
        )
    serve_inspection(
        session,
        port=args.port,
        review_path=args.save_review or args.review_file or Path("review.json"),
        output_dir=args.output_dir,
        protected_paths=protected,
        export_protected_paths=[args.review_file] if args.review_file is not None else (),
        spectrum_label=args.label,
        open_browser=not args.no_browser,
    )
    return 0


def _run_replay(args) -> int:
    from .review_exports import write_review_outputs

    session = _restore_review(args.spectrum, args.review_file.expanduser())
    result = session.reviewed
    write_review_outputs(
        session,
        args.output_dir,
        overwrite=args.overwrite,
        plot_format=args.plot_format,
        spectrum_label=args.label,
        protected_paths=[args.spectrum, args.review_file],
    )
    if result.temperature is None:
        print("Reviewed temperature is missing: no valid lines have been explicitly accepted.")
    else:
        print(
            f"Reviewed Teff = {result.temperature.temperature_k:.6f} K "
            f"from {result.temperature.n_lines} accepted lines"
        )
        print(
            f"Exact-subset empirical validation: N={result.performance.n_stars}, "
            f"bias={result.performance.bias_k:.6f} K, RMSE={result.performance.rmse_k:.6f} K"
        )
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
        return {"fit": _run_fit, "fit-hfs": _run_hfs,
                "inspect": _run_inspect, "replay": _run_replay}[args.command](args)
    except (ValueError, TypeError, OSError, ImportError) as exc:
        print(f"sc-teff: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
