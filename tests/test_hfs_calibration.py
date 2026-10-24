"""Frozen aggregate HFS numbers only; no spectra or private repository dependency."""

import csv
import hashlib
import json
import math
import statistics
from dataclasses import FrozenInstanceError
from importlib.resources import files

import pytest

from sc_teff_method import CALIBRATION_ID, LINE_IDS, get_calibration
from sc_teff_method import hfs_calibration as hfs

ROOT = files("sc_teff_method").joinpath("data", CALIBRATION_ID + "-hfs")
HASHES = {
    "calibration.json": "b62db1c8b3700cbd082664eda4e98c65a6f76fd3ea6113ef34d14395eb184ed3",
    "subset_metrics.csv": "a9e43b571d78365be83f96702212c1b688ff5207398f1d98fff8d4f887a9c906",
    "provenance.json": "f838b0eadba36ff5d26df2edc291a719424bc1acd635ca5b970929e0a1e68833",
}
VALIDATION_ROWS = list(csv.DictReader(ROOT.joinpath("subset_metrics.csv").read_text().splitlines()))


@pytest.mark.parametrize("name,expected", HASHES.items())
def test_frozen_hfs_products_are_hash_pinned(name, expected):
    assert hashlib.sha256(ROOT.joinpath(name).read_bytes()).hexdigest() == expected


def test_provenance_exports_only_two_hfs_models_and_four_allowed_aggregate_sources():
    manifest = json.loads(ROOT.joinpath("provenance.json").read_text())
    assert manifest["measurement_source"] == "hfs"
    assert manifest["calibration_id"] == CALIBRATION_ID
    assert manifest["snapshot_git_revision"] == "1b748b220ec03aa9b6417ad3737f7542436672de"
    assert {name.rsplit("/", 1)[-1] for name in manifest["source_sha256"]} == {
        "coefficients_all_methods.csv", "subset_metrics.csv", "line_quality.csv", "metadata.json",
    }
    assert manifest["export"]["coefficient_rows"] == 18
    assert manifest["export"]["validation_rows"] == 4088
    assert "Gaussian" in manifest["export"]["selection_anchor"]
    assert manifest["derived_sha256"] == {key: HASHES[key] for key in (
        "calibration.json", "subset_metrics.csv",
    )}


@pytest.mark.parametrize("configuration", ["nandakumar_feh", "nandakumar_no_feh"])
def test_calibration_identity_ranges_and_immutability(configuration):
    calibration = hfs.get_hfs_calibration(configuration)
    gaussian = get_calibration(configuration)
    assert calibration.measurement_source == "hfs"
    assert calibration.model_id == gaussian.model_id  # frozen IDs require the method discriminator
    assert calibration.selection_anchor_source == "gaussian"
    assert calibration.atomic_component_dataset == "hala2022_ab_generated"
    assert calibration.declared_strength_min_angstrom == 0.08
    assert calibration.declared_strength_max_angstrom is None
    assert tuple(calibration.lines) == LINE_IDS
    assert calibration.lines["line3"].n_training == 43
    assert calibration.lines["line1"].n_training == 44
    assert calibration.lines["line1"].a_k_per_angstrom != (
        gaussian.lines["line1"].a_k_per_angstrom
    )
    for line in LINE_IDS:
        assert calibration.lines[line].wavelength_air_angstrom == (
            gaussian.lines[line].wavelength_air_angstrom
        )
        assert (calibration.lines[line].b_k_per_dex is not None) == calibration.requires_feh
    with pytest.raises(FrozenInstanceError):
        calibration.model_id = "modified"
    with pytest.raises(TypeError):
        calibration.lines["line1"] = calibration.lines["line2"]


@pytest.mark.parametrize("configuration,expected_mean,expected_scatter", [
    ("nandakumar_feh", 3717.9335165929056, 117.93019844625434),
    ("nandakumar_no_feh", 3686.0148701121757, 85.90112716772033),
])
def test_fixed_hfs_temperature_regression_from_source_coefficients(
    configuration, expected_mean, expected_scatter,
):
    areas = {line: 0.22+0.019*i for i, line in enumerate(LINE_IDS)}
    result = hfs.estimate_hfs_temperature(areas, configuration=configuration, feh=-0.3)
    assert result.temperature_k == pytest.approx(expected_mean, abs=1e-9)
    assert result.line_scatter_k == pytest.approx(expected_scatter, abs=1e-9)
    assert result.measurement_source == "hfs"
    assert result.used_line_ids == LINE_IDS
    assert result.strict_strength_domain_applicable is None
    assert all(w.code == "strict_strength_domain_unknown" or "range" in w.code
               for w in result.warnings)


def test_individual_line_and_combined_hfs_regression_are_not_gaussian_predictions():
    areas = {"line2": 0.27, "line1": 0.30}
    result = hfs.estimate_hfs_temperature(areas, feh=-0.2)
    assert result.used_line_ids == ("line1", "line2")
    assert result.line_temperatures_k["line1"] == pytest.approx(3617.401241184987, abs=1e-9)
    assert result.line_temperatures_k["line2"] == pytest.approx(3668.3449182218806, abs=1e-9)
    assert result.temperature_k == pytest.approx(3642.873079703434, abs=1e-9)
    assert result.line_scatter_k == pytest.approx(36.02261949136473, abs=1e-9)
    single = hfs.estimate_hfs_temperature({"line2": 0.27}, feh=-0.2)
    assert single.line_scatter_k is None
    assert single.n_lines == 1


@pytest.mark.parametrize("configuration", ["nandakumar_feh", "nandakumar_no_feh"])
@pytest.mark.parametrize("mask", range(1, 512))
def test_all_511_hfs_subset_temperature_means(configuration, mask):
    selected = tuple(line for i, line in enumerate(LINE_IDS) if mask & (1 << i))
    areas = {line: 0.25+0.01*i for i, line in enumerate(selected)}
    calibration = hfs.get_hfs_calibration(configuration)
    temperatures = []
    for line, area in areas.items():
        row = calibration.lines[line]
        prediction = row.a_k_per_angstrom*area+row.c_k
        if row.b_k_per_dex is not None:
            prediction += row.b_k_per_dex*(-0.2)
        temperatures.append(prediction)
    result = hfs.estimate_hfs_temperature(
        areas, configuration=configuration, feh=-0.2,
        gaussian_anchor_areas={line: 0.1 for line in selected},
    )
    assert result.used_line_ids == selected
    assert result.temperature_k == statistics.mean(temperatures)
    assert result.strict_strength_domain_applicable is True


@pytest.mark.parametrize("row", VALIDATION_ROWS, ids=lambda row: "-".join(
    row[k] for k in ("configuration", "scope", "cohort", "subset_mask")
))
def test_every_frozen_hfs_subset_identity_and_statistic(row):
    mask = int(row["subset_mask"])
    selected = tuple(line for i, line in enumerate(LINE_IDS) if mask & (1 << i))
    result = hfs.hfs_subset_performance(
        reversed(selected), configuration=row["configuration"],
        scope=row["scope"], cohort=row["cohort"],
    )
    assert result.line_ids == selected
    assert result.n_stars == int(row["n_stars"])
    assert result.bias_k == float(row["bias_k"])
    assert result.rmse_k == float(row["rmse_k"])
    assert result.measurement_source == "hfs"
    assert result.calibration_id == CALIBRATION_ID
    assert result.selection_anchor_source == "gaussian"
    assert "Gaussian" in result.selection_note and "not HFS" in result.selection_note


def test_validation_inventory_is_all_511_subsets_in_all_eight_groups():
    identities = {(r["configuration"], r["scope"], r["cohort"], int(r["subset_mask"]))
                  for r in VALIDATION_ROWS}
    assert len(VALIDATION_ROWS) == len(identities) == 4088
    assert identities == {(c, s, h, m) for c in ("nandakumar_feh", "nandakumar_no_feh")
                          for s in ("strict_strength_domain", "all_available")
                          for h in ("exact_subset", "common_all9") for m in range(1, 512)}


@pytest.mark.parametrize("anchors,expected", [
    (None, None), ({}, None), ({"line1": 0.09}, None),
    ({"line1": 0.08, "line2": 0.08}, True),
    ({"line1": 0.079, "line2": 0.10}, False), ({"line1": 0.01}, False),
])
def test_strict_domain_uses_separate_gaussian_anchors_not_hfs_areas(anchors, expected):
    # These HFS areas are both below 0.08; a Gaussian-anchor pass is still a pass
    # of the declared selection, independently of HFS empirical-range warnings.
    result = hfs.estimate_hfs_temperature(
        {"line1": 0.06, "line2": 0.07}, feh=-0.2, gaussian_anchor_areas=anchors,
    )
    assert result.strict_strength_domain_applicable is expected
    assert all(w.code != "outside_declared_strength_cut" for w in result.warnings)
    assert any(w.code == "outside_training_strength_range" for w in result.warnings)
    if anchors is None or not anchors:
        assert any(w.code == "strict_strength_domain_unknown" for w in result.warnings)


def test_anchor_values_only_change_applicability_never_hfs_predictions():
    areas = {"line1": 0.3, "line2": 0.4}
    first = hfs.estimate_hfs_temperature(areas, feh=-0.2)
    second = hfs.estimate_hfs_temperature(
        areas, feh=-0.2, gaussian_anchor_areas={"line1": 0.1, "line2": 0.01},
    )
    assert first.temperature_k == second.temperature_k
    assert first.line_temperatures_k == second.line_temperatures_k
    assert first.strict_strength_domain_applicable is None
    assert second.strict_strength_domain_applicable is False
    with pytest.raises(TypeError):
        second.gaussian_anchor_areas["line1"] = 0.2


@pytest.mark.parametrize("value", [0, -0.1, math.nan, math.inf, -math.inf, True, None, "0.3"])
def test_invalid_hfs_areas_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        hfs.estimate_hfs_temperature({"line1": value}, feh=-0.2)


@pytest.mark.parametrize("areas", [{}, {"line10": 0.3}, {1: 0.3}, [], None, "line1"])
def test_missing_unknown_and_empty_hfs_measurements_rejected(areas):
    with pytest.raises((ValueError, TypeError)):
        hfs.estimate_hfs_temperature(areas, feh=-0.2)


@pytest.mark.parametrize("anchors", [
    {"line10": 0.1}, {"line1": 0}, {"line1": -1}, {"line1": math.nan},
    {"line1": math.inf}, {"line1": None}, {"line1": True}, [], "0.3",
])
def test_invalid_gaussian_anchor_measurements_rejected(anchors):
    with pytest.raises((ValueError, TypeError)):
        hfs.estimate_hfs_temperature({"line1": 0.3}, feh=-0.2, gaussian_anchor_areas=anchors)


def test_method_tagged_measurements_cannot_cross_the_hfs_api():
    class GaussianAreas(dict):
        measurement_source = "gaussian"

    class HFSAreas(dict):
        measurement_source = "hfs"

    with pytest.raises(ValueError, match="HFS calibration requires HFS"):
        hfs.estimate_hfs_temperature(GaussianAreas(line1=0.3), feh=-0.2)
    with pytest.raises(ValueError, match="requires Gaussian"):
        hfs.estimate_hfs_temperature(
            HFSAreas(line1=0.3), feh=-0.2, gaussian_anchor_areas=HFSAreas(line1=0.3),
        )


def test_missing_metallicity_and_no_feh_independence():
    with pytest.raises(ValueError, match="requires an explicit"):
        hfs.estimate_hfs_temperature({"line1": 0.3})
    first = hfs.estimate_hfs_temperature({"line1": 0.3}, configuration="nandakumar_no_feh")
    second = hfs.estimate_hfs_temperature(
        {"line1": 0.3}, configuration="nandakumar_no_feh", feh=5,
    )
    assert first.temperature_k == second.temperature_k
    assert any(w.code == "outside_reference_feh_range" for w in second.warnings)


def test_external_temperature_and_metallicity_ranges_remain_diagnostic():
    result = hfs.estimate_hfs_temperature(
        {"line3": 0.3}, feh=-1.0, reference_temperature_k=6000,
    )
    assert result.used_line_ids == ("line3",)
    assert {w.code for w in result.warnings} >= {
        "outside_reference_temperature_range", "outside_reference_feh_range",
        "strict_strength_domain_unknown",
    }


@pytest.mark.parametrize("kwargs", [
    {"configuration": "gaussian"}, {"scope": "unknown"}, {"cohort": "unknown"},
])
def test_unknown_hfs_validation_selectors_rejected(kwargs):
    with pytest.raises(ValueError):
        hfs.hfs_subset_performance(["line1"], **kwargs)


@pytest.mark.parametrize("lines", [[], ["line1", "line1"], ["line10"], "line1", None])
def test_invalid_hfs_subset_ids_rejected(lines):
    with pytest.raises((ValueError, TypeError)):
        hfs.hfs_subset_performance(lines)


@pytest.mark.parametrize("filename", ["provenance.json", "calibration.json", "subset_metrics.csv"])
def test_hfs_data_integrity_rejects_tampering(tmp_path, monkeypatch, filename):
    for name in HASHES:
        (tmp_path / name).write_bytes(ROOT.joinpath(name).read_bytes())
    with (tmp_path / filename).open("ab") as stream:
        stream.write(b" ")
    hfs._manifest.cache_clear()
    monkeypatch.setattr(hfs, "_DATA_ROOT", tmp_path)
    try:
        with pytest.raises(RuntimeError, match="SHA-256"):
            hfs._read_verified("subset_metrics.csv" if filename == "provenance.json" else filename)
    finally:
        hfs._manifest.cache_clear()
