"""Effective temperatures from separate frozen Sc I Gaussian and HFS calibrations."""

from importlib import import_module

from ._data import CALIBRATION_ID, COHORTS, CONFIGURATIONS, LINE_IDS, SCOPES, get_calibration
from .api import estimate_temperature, subset_performance
from .hfs_calibration import (
    estimate_hfs_temperature,
    get_hfs_calibration,
    hfs_subset_performance,
)
from .measurements import LineAreas
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
    "ContinuumCorrection": "continuum", "ContinuumOptions": "continuum",
    "evaluate_continuum": "continuum", "apply_continuum": "continuum",
    "HFSLevel": "hfs", "HFSComponent": "hfs", "hfs_absorption": "hfs", "hfs_area": "hfs",
    "load_hfs_atomic_data": "hfs_data",
    "HFSParameters": "hfs_fitting", "HFSFitOptions": "hfs_fitting",
    "HFSLineFit": "hfs_fitting", "fit_hfs_lines": "hfs_fitting",
    "HFSSpectralEstimate": "hfs_workflow", "estimate_hfs_from_spectrum": "hfs_workflow",
    "fit_hfs_spectrum": "hfs_workflow", "plot_hfs_diagnostics": "hfs_outputs",
    "write_hfs_outputs": "hfs_outputs",
    "HFSReviewSession": "hfs_review", "read_hfs_review": "hfs_review",
    "write_hfs_review_outputs": "hfs_review_exports",
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
    "LineAreas",
    "get_hfs_calibration",
    "estimate_hfs_temperature",
    "hfs_subset_performance",
]

# Preserve dependency-free wildcard imports from PR 1. Spectral APIs are
# available through explicit imports, which load the optional libraries lazily.
