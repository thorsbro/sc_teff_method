"""Anchor the frozen bytes and reject edited package data before using it."""

import hashlib
import json
from importlib.resources import files

import pytest

from sc_teff_method import CALIBRATION_ID, CONFIGURATIONS, LINE_IDS, _data, get_calibration

EXPECTED = {
    "calibration.json": "5aacb112f19ca739ba8c973622e77eff7f9e424095699bb8a59e7d6fee309c84",
    "subset_metrics.csv": "2c7618489d2849afb7d2e93899e607500427f8b62dc50b63bd4863d074b8231c",
    "provenance.json": "f9afdda035971d77a998ed00a82cd0ea7ada23cf720026905206b1b235c40cb6",
}
DATA = files("sc_teff_method").joinpath("data", CALIBRATION_ID)


def test_frozen_coefficient_validation_and_provenance_integrity():
    assert {p.name for p in DATA.iterdir()} == set(EXPECTED)
    for name, digest in EXPECTED.items():
        assert hashlib.sha256(DATA.joinpath(name).read_bytes()).hexdigest() == digest
    manifest = json.loads(DATA.joinpath("provenance.json").read_bytes())
    assert manifest["calibration_id"] == CALIBRATION_ID
    assert manifest["export"]["coefficient_rows"] == 18
    assert manifest["export"]["validation_rows"] == 4088
    expected_derived = {k: v for k, v in EXPECTED.items() if k != "provenance.json"}
    assert manifest["derived_sha256"] == expected_derived
    for configuration in CONFIGURATIONS:
        calibration = get_calibration(configuration)
        assert tuple(calibration.lines) == LINE_IDS
        for line_id, line in calibration.lines.items():
            assert line.n_training == (43 if line_id == "line3" else 44)
            assert line.a_k_per_angstrom < 0
            assert (line.b_k_per_dex is not None) == calibration.requires_feh
            assert line.training_strength_range_angstrom[0] > 0.08
            lower, upper = line.training_strength_range_angstrom
            assert lower < upper


@pytest.fixture
def copied_data(tmp_path, monkeypatch):
    for name in EXPECTED:
        tmp_path.joinpath(name).write_bytes(DATA.joinpath(name).read_bytes())
    monkeypatch.setattr(_data, "_DATA_ROOT", tmp_path)
    _data._manifest.cache_clear()
    _data._calibrations.cache_clear()
    _data._subset_index.cache_clear()
    yield tmp_path
    _data._manifest.cache_clear()
    _data._calibrations.cache_clear()
    _data._subset_index.cache_clear()


@pytest.mark.parametrize("name", EXPECTED)
def test_edited_calibration_or_validation_is_rejected(copied_data, name):
    target = copied_data / name
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="SHA-256 integrity"):
        if name == "subset_metrics.csv":
            _data._subset_index()
        else:
            get_calibration()
