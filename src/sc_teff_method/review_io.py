"""Shared spectrum loading and byte identity for local fit/review/replay commands."""

from __future__ import annotations

import hashlib
from pathlib import Path

LOADER_DEFAULTS = {
    "wavelength_column": 0,
    "flux_column": 1,
    "uncertainty_column": None,
    "mask_column": None,
    "delimiter": None,
    "header": False,
    "skiprows": 0,
}


def load_spectrum_identity(
    path: str | Path,
    *,
    wavelength_unit: str,
    wavelength_frame: str,
    continuum_normalised: bool,
    loader_settings: dict | None = None,
    rv_kms: float | None = None,
):
    """Use the PR 2 loader, detecting source-file changes during loading.

    The returned portable identity has a basename and SHA-256, never a local
    absolute path or embedded spectrum. Coordinates are corrected exactly once.
    """
    from .spectrum import load_spectrum

    source = Path(path).expanduser()
    settings = dict(LOADER_DEFAULTS)
    supplied = loader_settings or {}
    unknown = set(supplied) - set(settings)
    if unknown:
        raise ValueError(f"Unknown spectrum loader settings: {sorted(unknown)}")
    settings.update(supplied)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    spectrum = load_spectrum(
        source,
        **settings,
        wavelength_unit=wavelength_unit,
        wavelength_frame=wavelength_frame,
        continuum_normalised=continuum_normalised,
        rv_kms=rv_kms,
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
        raise ValueError("Input spectrum changed during loading; load a stable copy")
    return spectrum, {"filename": source.name, "sha256": checksum, "loader_settings": settings}


def protect_output_inputs(output_dir, protected_paths, *, include_review: bool = True) -> None:
    """Refuse replacing source/configuration paths, including expanded aliases."""
    destination = Path(output_dir).expanduser()
    names = ["summary.json", "lines.csv", "diagnostics.png", "diagnostics.pdf"]
    if include_review:
        names.append("review.json")
    sources = [Path(path).expanduser().resolve() for path in protected_paths if path is not None]
    if any((destination / name).resolve() in sources for name in names):
        raise ValueError(
            "The output would replace an input spectrum or configuration/review; "
            "choose another directory"
        )
