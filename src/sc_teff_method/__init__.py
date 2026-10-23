"""Effective temperatures from frozen Sc I Gaussian line-area calibrations."""

from importlib import import_module

from ._data import CALIBRATION_ID, COHORTS, CONFIGURATIONS, LINE_IDS, SCOPES, get_calibration
from .api import estimate_temperature, subset_performance
from .models import (
    ApplicabilityWarning,
    Calibration,
    LineCalibration,
    SubsetPerformance,
    TemperatureEstimate,
)

__version__ = "0.1.0"

# Keep PR 1's measured-area API usable without the optional spectral libraries.
_SPECTRAL_EXPORTS = {
    "Spectrum": "spectrum", "prepare_spectrum": "spectrum", "load_spectrum": "spectrum",
    "GaussianParameters": "fitting", "FitOptions": "fitting", "LineFit": "fitting",
    "gaussian_absorption": "fitting", "gaussian_area": "fitting", "fit_gaussian_lines": "fitting",
    "SpectralEstimate": "workflow", "fit_spectrum": "workflow",
    "estimate_from_spectrum": "workflow", "plot_diagnostics": "diagnostics",
    "write_fit_outputs": "outputs",
    "ReviewSession": "review", "read_review": "review",
    "write_review_outputs": "review_exports",
}


def __getattr__(name: str):
    if name not in _SPECTRAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(f".{_SPECTRAL_EXPORTS[name]}", __name__), name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"numpy", "scipy", "matplotlib"}:
            raise ImportError(
                "Spectrum fitting needs the optional libraries: "
                "install 'sc-teff-method[spectra]'"
            ) from exc
        raise
    globals()[name] = value
    return value

__all__ = [
    "CALIBRATION_ID",
    "COHORTS",
    "CONFIGURATIONS",
    "LINE_IDS",
    "SCOPES",
    "ApplicabilityWarning",
    "Calibration",
    "LineCalibration",
    "SubsetPerformance",
    "TemperatureEstimate",
    "estimate_temperature",
    "get_calibration",
    "subset_performance",
]

# Preserve dependency-free wildcard imports from PR 1. Spectral APIs are
# available through explicit imports, which load the optional libraries lazily.
