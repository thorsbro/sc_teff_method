"""Exhaust every published subset identity, model, scope and cohort."""

import csv
import itertools
from importlib.resources import files

import pytest

from sc_teff_method import (
    CALIBRATION_ID,
    COHORTS,
    CONFIGURATIONS,
    LINE_IDS,
    SCOPES,
    subset_performance,
)

with files("sc_teff_method").joinpath("data", CALIBRATION_ID, "subset_metrics.csv").open() as f:
    ROWS = list(csv.DictReader(f))


@pytest.mark.parametrize("row", ROWS, ids=lambda r: "-".join(
    r[k] for k in ("configuration", "scope", "cohort", "subset_mask")
))
def test_every_published_subset_lookup(row):
    mask = int(row["subset_mask"])
    chosen = tuple(line for i, line in enumerate(LINE_IDS) if mask & (1 << i))
    result = subset_performance(
        iter(reversed(chosen)), configuration=row["configuration"],
        scope=row["scope"], cohort=row["cohort"],
    )
    assert result.line_ids == chosen
    assert result.n_lines == len(chosen)
    assert result.n_stars == int(row["n_stars"])
    assert result.bias_k == float(row["bias_k"])
    assert result.rmse_k == float(row["rmse_k"])
    assert result.scope == row["scope"]
    assert result.cohort == row["cohort"]
    assert result.configuration == row["configuration"]
    assert result.calibration_id == CALIBRATION_ID
    assert result.temperature_scale_id == "nandakumar2023_igrins"
    assert result.measurement_source == "gaussian"
    assert result.model_id == (
        "legacy_direct_metallicity_0p08" if row["configuration"] == "nandakumar_feh"
        else "legacy_direct_1d_0p08"
    )
    expected_n = 43 if row["cohort"] == "common_all9" or "line3" in chosen else 44
    assert result.n_stars == expected_n
    assert "not formal measurement uncertainties" in result.uncertainty_note


def test_all_511_combinatorial_identities_in_each_group():
    expected = {chosen for n in range(1, 10) for chosen in itertools.combinations(LINE_IDS, n)}
    assert len(ROWS) == 4088
    assert len(expected) == 511
    for configuration, scope, cohort in itertools.product(CONFIGURATIONS, SCOPES, COHORTS):
        selected = [r for r in ROWS if (r["configuration"], r["scope"], r["cohort"])
                    == (configuration, scope, cohort)]
        masks = [int(r["subset_mask"]) for r in selected]
        assert len(masks) == len(set(masks)) == 511
        found = {
            tuple(line for i, line in enumerate(LINE_IDS) if mask & (1 << i)) for mask in masks
        }
        assert found == expected


@pytest.mark.parametrize("configuration,n,bias,rmse", [
    ("nandakumar_feh", 44, -1.0468992999376239, 51.379990357673506),
    ("nandakumar_no_feh", 44, -0.13726696487494586, 57.795114330075037),
])
def test_fixed_validation_regressions_and_bias_sign(configuration, n, bias, rmse):
    result = subset_performance(["line1", "line2"], configuration=configuration)
    assert result.n_stars == n
    assert result.bias_k == pytest.approx(bias, rel=0, abs=1e-12)
    assert result.rmse_k == pytest.approx(rmse, rel=0, abs=1e-12)


def test_common_cohort_remains_distinct_from_exact_subset():
    exact = subset_performance(["line1"])
    common = subset_performance(["line1"], cohort="common_all9")
    assert (exact.n_stars, common.n_stars) == (44, 43)
    assert exact.rmse_k != common.rmse_k
    nine = subset_performance(LINE_IDS)
    assert nine.n_lines == 9
    assert nine.n_stars == 43
    assert nine.rmse_k == pytest.approx(48.9524, abs=0.00005)


def test_strict_and_all_available_have_distinct_identity_even_when_values_agree():
    strict = subset_performance(LINE_IDS)
    available = subset_performance(LINE_IDS, scope="all_available")
    assert strict.scope != available.scope
    assert strict.n_stars == available.n_stars
    assert strict.rmse_k == available.rmse_k


@pytest.mark.parametrize("line_ids", [[], (), ["line1", "line1"], ["line10"], [None], [[]]])
def test_invalid_subsets(line_ids):
    with pytest.raises(ValueError):
        subset_performance(line_ids)


@pytest.mark.parametrize("line_ids", [None, "line1", b"line1", 1])
def test_invalid_subset_containers(line_ids):
    with pytest.raises(TypeError):
        subset_performance(line_ids)


@pytest.mark.parametrize("keyword,value", [
    ("configuration", "apogee_mh"), ("configuration", None), ("configuration", []),
    ("scope", "training_domain"), ("scope", None), ("scope", []),
    ("cohort", "best"), ("cohort", None), ("cohort", []),
])
def test_unknown_validation_options(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        subset_performance(["line1"], **{keyword: value})
