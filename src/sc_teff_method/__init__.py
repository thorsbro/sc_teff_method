"""Effective temperatures from frozen Sc I Gaussian line-area calibrations."""

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
