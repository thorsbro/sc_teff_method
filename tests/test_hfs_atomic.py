"""Atomic equations and synthetic HFS profiles, without private repository access."""
# ruff: noqa: E402

import copy
import csv
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
from types import MappingProxyType, ModuleType

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from scipy.integrate import trapezoid

from sc_teff_method.hfs import (
    HFSComponent,
    HFSLevel,
    allowed_f_values,
    allowed_hfs_transitions,
    fit_hfs_profile,
    generate_hfs_components,
    hfs_absorption,
    hfs_area,
    hfs_relative_strength,
    hyperfine_energy_shift,
    wavenumber_offset_to_wavelength_offset,
)
from sc_teff_method.hfs_data import (
    _DATA_ROOT,
    ATOMIC_ID,
    COMPONENT_COUNTS,
    HFS_DATASET_ID,
    _decode_atomic,
    load_hfs_atomic_data,
)


@pytest.mark.parametrize("spin,j,expected", [
    (0, 1, (1,)), (0.5, 0.5, (0, 1)), (1, 1, (0, 1, 2)),
    (3.5, 0.5, (3, 4)), (3.5, 1.5, (2, 3, 4, 5)),
    (3.5, 4.5, tuple(range(1, 9))), (1, 1.5, (0.5, 1.5, 2.5)),
])
def test_angular_coupling_values(spin, j, expected):
    assert allowed_f_values(spin, j) == expected


@pytest.mark.parametrize("bad", [-1, 0.1, 0.75, float("nan"), float("inf"), True, None, "1"])
def test_invalid_angular_momenta_fail(bad):
    with pytest.raises(ValueError):
        allowed_f_values(bad, 1)
    with pytest.raises(ValueError):
        HFSLevel(bad, 0)


def test_selection_rules_and_exact_known_wigner_strengths():
    assert allowed_hfs_transitions(0.5, 0.5, 0.5) == ((0, 1), (1, 0), (1, 1))
    assert allowed_hfs_transitions(1, 0, 0) == ()
    assert allowed_hfs_transitions(0, 1, 1.5) == ()
    assert allowed_hfs_transitions(3.5, 0.5, 3.5) == ()
    generated = generate_hfs_components(0.5, HFSLevel(0.5, 0), HFSLevel(0.5, 0))
    assert {(c.F_lower, c.F_upper): c.strength for c in generated} == {
        (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.5,
    }
    assert hfs_relative_strength(0.5, 0.5, 0.5, 0, 0) == 0
    assert hfs_relative_strength(3.5, 1.5, 1.5, 3, 3) == 0
    assert len(allowed_hfs_transitions(3.5, 1.5, 1.5)) == 10
    assert len(generate_hfs_components(3.5, HFSLevel(1.5, 1), HFSLevel(1.5, 2))) == 9
    with pytest.raises(ValueError):
        HFSComponent(0, 0, 0, 1)
    with pytest.raises(ValueError):
        HFSComponent(1, 1.5, 0, 1)
    with pytest.raises(ValueError):
        hfs_relative_strength(0.5, 0.5, 0.5, 2, 1)


def independent_energy(spin, j, f, a, b):
    k = f*(f+1)-spin*(spin+1)-j*(j+1)
    dipole = a*k/2
    if spin < 1 or j < 1:
        return dipole
    return dipole+b*(3*k*(k+1)-4*spin*(spin+1)*j*(j+1))/(8*spin*(2*spin-1)*j*(2*j-1))


def test_level_energy_and_weighted_centroid_independent_expression():
    assert hyperfine_energy_shift(1, 1, 2, 4, 8) == 6
    assert hyperfine_energy_shift(1, 1, 0, 4, 8) == 12
    assert hyperfine_energy_shift(0.5, 1.5, 1, 2, 100) == (
        hyperfine_energy_shift(0.5, 1.5, 1, 2)
    )
    for record in load_hfs_atomic_data().levels.values():
        level = record.level
        shifts = []
        for f in allowed_f_values(3.5, level.J):
            actual = hyperfine_energy_shift(3.5, level.J, f, level.A, level.B)
            assert actual == pytest.approx(
                independent_energy(3.5, level.J, f, level.A, level.B), abs=1e-16,
            )
            shifts.append((2*f+1)*actual)
        assert math.fsum(shifts) == pytest.approx(0, abs=2e-15)


def test_atomic_inventory_frozen_strength_definition_and_immutability():
    data = load_hfs_atomic_data()
    assert data.policy.policy_id == ATOMIC_ID
    assert data.dataset_id == HFS_DATASET_ID == "hala2022_ab_generated"
    assert data.policy.I == 3.5
    assert tuple(map(len, data.components.values())) == (21, 16, 9, 12, 21, 18, 6, 12, 18)
    assert sum(map(len, data.components.values())) == 133
    assert len(data.levels) == 10
    assert {key: len(value) for key, value in data.components.items()} == COMPONENT_COUNTS
    assert all(sum(c.strength for c in pattern) == 1 for pattern in data.components.values())
    for key in data.lines:
        assert sum(data.raw_strengths[key]) == pytest.approx(
            1.000001 if key == "line1" else 0.999999 if key == "line5" else 1,
        )
    with pytest.raises(TypeError):
        data.lines["line1"] = None
    with pytest.raises(TypeError):
        data.levels["upper_2D5_2"].source["B_status"] = "unknown"
    with pytest.raises(ValueError):
        load_hfs_atomic_data("hala2022_table4_published")
    with pytest.raises(ValueError):
        load_hfs_atomic_data("vandeelen2017_legacy")


@pytest.mark.parametrize("key", tuple(COMPONENT_COUNTS))
def test_all_133_component_assignments_offsets_and_wigner_rounding(key):
    data = load_hfs_atomic_data()
    line = data.lines[key]
    lower, upper = (data.levels[name].level for name in (line.lower_level_id, line.upper_level_id))
    theory = {(c.F_lower, c.F_upper): c for c in generate_hfs_components(3.5, lower, upper)}
    assert len(theory) == COMPONENT_COUNTS[key]
    for c, rounded in zip(data.components[key], data.raw_strengths[key], strict=True):
        expected = theory[c.F_lower, c.F_upper]
        assert c.offset_cm1 == pytest.approx(
            independent_energy(3.5, upper.J, c.F_upper, upper.A, upper.B)
            - independent_energy(3.5, lower.J, c.F_lower, lower.A, lower.B), abs=5.01e-13,
        )
        assert rounded == pytest.approx(expected.strength, abs=5.01e-7)
    # The rounded published pattern deliberately differs from exact Wigner strengths.
    if key != "line3":
        assert any(abs(c.strength-theory[c.F_lower, c.F_upper].strength) > 1e-8
                   for c in data.components[key])


def test_table4_a_only_distinction_positive_line9_and_missing_b_statuses():
    data = load_hfs_atomic_data()
    a_only_max = line9_ab = 0.0
    for key, line in data.lines.items():
        lower, upper = (data.levels[name].level
                        for name in (line.lower_level_id, line.upper_level_id))
        for c, reference in zip(data.components[key], data.published_components[key], strict=True):
            expected_a = independent_energy(3.5, upper.J, c.F_upper, upper.A, 0)
            expected_a -= independent_energy(3.5, lower.J, c.F_lower, lower.A, 0)
            a_only_max = max(a_only_max, abs(reference.offset_cm1-expected_a))
            if key == "line9":
                line9_ab = max(line9_ab, abs(c.offset_cm1-reference.offset_cm1))
    assert a_only_max == pytest.approx(2.36e-5, abs=1e-12)
    assert line9_ab == pytest.approx(0.008399767857, abs=1e-12)
    assert data.levels["upper_2D5_2"].level.A == 0.02205
    assert data.levels["upper_2D5_2"].level.B == -0.01581
    assert data.levels["upper_4D1_2"].B_status == "not_applicable_J_lt_1"
    for key in ("upper_4D3_2", "upper_4D5_2", "upper_2D3_2"):
        assert data.levels[key].B_status == "unreported_assumed_zero"
        assert data.levels[key].level.B == 0
        assert data.levels[key].source["B_uncertainty_cm1"] is None


@pytest.mark.parametrize("offset", [0, 0.1, -0.1, 0.3, 1e-12, -1e-12, 1e-25])
def test_exact_wavelength_conversion_against_high_precision_decimal(offset):
    parent, cog = 21730.467, 4600.57
    with localcontext() as context:
        context.prec = 80
        p, w, d = Decimal(str(parent)), Decimal(str(cog)), Decimal(str(offset))
        expected = float(p*w/(w+d)-p)
    result = float(wavenumber_offset_to_wavelength_offset(offset, parent, cog))
    assert result == pytest.approx(expected, rel=6e-16, abs=1e-40)
    if offset:
        assert math.copysign(1, result) == -math.copysign(1, offset)
    else:
        assert result == 0
    if offset == 0.3:
        linear_approximation = -parent*offset/cog
        assert abs(result-linear_approximation) > 1e-5
        wrong_air_cog = float(wavenumber_offset_to_wavelength_offset(offset, parent, 1e8/parent))
        assert abs(result-wrong_air_cog) > 0.0001


@pytest.mark.parametrize("bad", [True, "1", None, float("nan"), float("inf")])
def test_nonfinite_or_nonreal_constants_and_profile_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        HFSLevel(1, bad)
    with pytest.raises(ValueError):
        HFSComponent(1, 1, bad, 1)
    with pytest.raises(ValueError):
        hfs_area(bad, 0.3)
    with pytest.raises(ValueError):
        wavenumber_offset_to_wavelength_offset(bad, 22000, 4500)


@pytest.mark.parametrize("amplitude,sigma", [(-0.1, 1), (1.1, 1), (0.1, 0), (0.1, -1)])
def test_invalid_profile_amplitude_or_width(amplitude, sigma):
    with pytest.raises(ValueError):
        hfs_area(amplitude, sigma)
    with pytest.raises(ValueError):
        hfs_absorption([22000], amplitude, 22000, sigma, (HFSComponent(0, 1, 0, 1),),
                       parent_wavelength=22000, cog_wavenumber_cm1=4500)


@pytest.mark.parametrize("key", tuple(COMPONENT_COUNTS))
def test_full_analytical_mixture_and_area_recovery(key):
    data = load_hfs_atomic_data()
    line, components = data.lines[key], data.components[key]
    parent = line.wavelength_air_angstrom
    kwargs = dict(parent_wavelength=parent, cog_wavenumber_cm1=line.cog_wavenumber_cm1)
    x = parent+np.linspace(-4, 4, 801)
    offsets = np.array([-parent*c.offset_cm1/(line.cog_wavenumber_cm1+c.offset_cm1)
                        for c in components])
    a, centre, sigma = 0.57, parent+0.09, 0.36
    independent = 1-a*np.sum(np.array([c.strength for c in components])
                            * np.exp(-0.5*((x[:, None]-centre-offsets)/sigma)**2), axis=1)
    np.testing.assert_allclose(hfs_absorption(x, a, centre, sigma, components, **kwargs),
                               independent, atol=3e-16, rtol=0)
    fit = fit_hfs_profile(x, independent, components, **kwargs)
    assert fit.status == "converged"
    assert fit.amplitude == pytest.approx(a, abs=2e-8)
    assert fit.center_angstrom == pytest.approx(centre, abs=2e-8)
    assert fit.sigma_angstrom == pytest.approx(sigma, abs=2e-8)
    assert fit.area_angstrom == pytest.approx(math.sqrt(2*math.pi)*a*sigma, abs=2e-8)
    wide = parent+np.linspace(-12, 12, 24001)
    integral = trapezoid(1-hfs_absorption(wide, a, centre, sigma, components, **kwargs), wide)
    assert integral == pytest.approx(hfs_area(a, sigma), rel=2e-12)
    assert fit.parameter_errors is not None and fit.covariance is not None


def test_full_covariance_and_area_error_against_independent_propagation():
    data = load_hfs_atomic_data()
    line, components = data.lines["line9"], data.components["line9"]
    parent, cog = line.wavelength_air_angstrom, line.cog_wavenumber_cm1
    x = parent+np.linspace(-3, 3, 501)
    kwargs = dict(parent_wavelength=parent, cog_wavenumber_cm1=cog)
    y = hfs_absorption(x, 0.56, parent+0.08, 0.34, components, **kwargs)
    y += np.random.default_rng(20261005).normal(0, 0.002, len(x))
    result = fit_hfs_profile(x, y, components, **kwargs)
    a, centre, sigma = result.amplitude, result.center_angstrom, result.sigma_angstrom
    offsets = np.array([-parent*c.offset_cm1/(cog+c.offset_cm1) for c in components])
    strengths = np.array([c.strength for c in components])
    d = x[:, None]-centre-offsets
    e = np.exp(-0.5*(d/sigma)**2)
    j = np.column_stack((-e@strengths, -a*(e*d)@strengths/sigma**2,
                         -a*(e*d*d)@strengths/sigma**3))
    residual = 1-a*(e@strengths)-y
    covariance = np.linalg.inv(j.T@j)*np.dot(residual, residual)/(len(x)-3)
    np.testing.assert_allclose(result.covariance, covariance, rtol=2e-9, atol=1e-15)
    expected = 2*math.pi*(sigma**2*covariance[0, 0]+a**2*covariance[2, 2]
                         +2*a*sigma*covariance[0, 2])
    assert result.area_formal_error_angstrom == pytest.approx(math.sqrt(expected), rel=2e-9)
    without_cross_term = 2*math.pi*(sigma**2*covariance[0, 0]+a**2*covariance[2, 2])
    assert abs(expected-without_cross_term) > expected*0.2


@pytest.mark.parametrize("problem", ["count", "negative_a", "b_status", "offset", "strength", "f"])
def test_modified_atomic_contents_fail_closed(problem):
    payload = json.loads(_DATA_ROOT.joinpath("atomic.json").read_text())
    provenance = json.loads(_DATA_ROOT.joinpath("atomic_provenance.json").read_text())
    if problem == "count":
        payload["lines"]["line1"]["components"].pop()
    elif problem == "negative_a":
        payload["levels"]["upper_2D5_2"]["A_cm1"] *= -1
    elif problem == "b_status":
        payload["levels"]["upper_4D3_2"]["B_status"] = "not_applicable_J_lt_1"
    else:
        field = {"offset": "offset_cm1", "strength": "strength_published", "f": "F_lower"}[problem]
        payload["lines"]["line1"]["components"][0][field] += 0.1
    with pytest.raises((ValueError, RuntimeError)):
        _decode_atomic(payload, provenance)


def test_packaged_data_hashes_and_rejected_integrity_failure(tmp_path, monkeypatch):
    from sc_teff_method import hfs_data

    manifest = json.loads(_DATA_ROOT.joinpath("atomic_provenance.json").read_text())
    raw = _DATA_ROOT.joinpath("atomic.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["derived_sha256"]["atomic.json"]
    assert manifest["publisher_table4_sha256"] == (
        "54a82bd6dfcdfed9b850a4a4ab60a3e1b8976a7b870266142db4dc3971c0dcb1"
    )
    assert len(json.loads(raw)["lines"]) == 9
    for name in ("atomic.json", "atomic_provenance.json"):
        (tmp_path/name).write_bytes(_DATA_ROOT.joinpath(name).read_bytes())
    changed = copy.deepcopy(json.loads(raw))
    changed["lines"]["line1"]["components"][0]["offset_cm1"] += 0.01
    (tmp_path/"atomic.json").write_text(json.dumps(changed))
    monkeypatch.setattr(hfs_data, "_DATA_ROOT", tmp_path)
    hfs_data.load_hfs_atomic_data.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="SHA-256"):
            hfs_data.load_hfs_atomic_data()
    finally:
        hfs_data.load_hfs_atomic_data.cache_clear()


def public_tool(name):
    path = Path(__file__).parents[1]/"tools"/f"{name}.py"
    module = ModuleType(f"test_{name}")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("alias", ["inside", "parent_symlink", "file_symlink", "hard_link"])
def test_atomic_export_rejects_reference_destinations_before_writing(tmp_path, alias):
    export_tool = public_tool("export_hfs_atomic")
    reference = tmp_path/"reference"
    reference.mkdir()
    for relative in export_tool.SOURCE_PATHS:
        path = reference/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic read-only sentinel\n")
    protected = reference/"config/hfs_levels.csv"
    original = protected.read_bytes()
    destination = tmp_path/"output"
    destination.mkdir()
    if alias == "inside":
        destination = reference/"output"
    elif alias == "parent_symlink":
        destination.rmdir()
        destination.symlink_to(reference, target_is_directory=True)
    elif alias == "file_symlink":
        (destination/"atomic.json").symlink_to(protected)
    else:
        (destination/"atomic.json").hardlink_to(protected)
    with pytest.raises(ValueError, match="read-only"):
        export_tool.export(reference, destination)
    assert protected.read_bytes() == original
    assert not (reference/"output").exists()


def test_reference_verifier_executes_only_verified_bytes_without_creating_private_cache(
    tmp_path, monkeypatch,
):
    """Only a temporary numerical source and public atomic facts are used here."""
    from dataclasses import replace

    verifier = public_tool("verify_hfs_atomic_reference")
    from sc_teff_method import hfs as public_hfs

    data = load_hfs_atomic_data()
    payload = json.loads(_DATA_ROOT.joinpath("atomic.json").read_text())
    reference = tmp_path/"numerical_reference"
    source = reference/"src/scteff/hfs.py"
    source.parent.mkdir(parents=True)
    public_source = Path(public_hfs.__file__)
    source.write_bytes(public_source.read_bytes())
    config = reference/"config"
    config.mkdir()
    for filename, fieldnames, rows in (
        ("hfs_levels.csv", ("dataset_id", "J", "A_model_value_cm1", "B_model_value_cm1"), [
            {"dataset_id": "hala2022_primary", "J": level["J"],
             "A_model_value_cm1": level["A_cm1"], "B_model_value_cm1": level["B_cm1"]}
            for level in payload["levels"].values()
        ]),
        ("hfs_components_hala2022_ab.csv", ("line_id", "F_lower", "F_upper",
                                            "component_offset_cm1", "strength"), [
            {"line_id": key, "F_lower": c["F_lower"], "F_upper": c["F_upper"],
             "component_offset_cm1": c["offset_cm1"], "strength": c["strength_frozen"]}
            for key, line in payload["lines"].items() for c in line["components"]
        ]),
    ):
        with (config/filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    paths = ("src/scteff/hfs.py", "config/hfs_levels.csv", "config/hfs_components_hala2022_ab.csv")
    hashes = {path: hashlib.sha256((reference/path).read_bytes()).hexdigest() for path in paths}
    monkeypatch.setattr(verifier, "load_hfs_atomic_data",
                        lambda: replace(data, source_hashes=MappingProxyType(hashes)))
    report = verifier.verify(reference)
    assert max(report["maximum_absolute_differences"].values()) == 0
    assert not (source.parent/"__pycache__").exists()
    actual_paths = {str(path.relative_to(reference))
                    for path in reference.rglob("*") if path.is_file()}
    assert actual_paths == set(paths)


def test_custom_condition_threshold_changes_flags_without_changing_fit():
    data = load_hfs_atomic_data()
    line, components = data.lines["line1"], data.components["line1"]
    x = line.wavelength_air_angstrom+np.linspace(-3, 3, 301)
    kwargs = dict(parent_wavelength=line.wavelength_air_angstrom,
                  cog_wavenumber_cm1=line.cog_wavenumber_cm1)
    y = hfs_absorption(x, 0.5, line.wavelength_air_angstrom, 0.35, components, **kwargs)
    first = fit_hfs_profile(x, y, components, max_jacobian_condition=1.0001, **kwargs)
    second = fit_hfs_profile(x, y, components, max_jacobian_condition=1e12, **kwargs)
    assert "ill_conditioned_covariance" in first.flags
    assert "ill_conditioned_covariance" not in second.flags
    assert first.area_angstrom == second.area_angstrom
    assert first.area_formal_error_angstrom == second.area_formal_error_angstrom
