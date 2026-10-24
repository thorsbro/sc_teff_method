"""Independent angular-momentum identities and invented analytical HFS profiles.

These checks use neither private modules nor observational templates. Their
inputs are invented level constants, exact coupling sum rules and seeded noise.
"""
# ruff: noqa: E402

import math
from itertools import product

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from scipy.integrate import trapezoid

from sc_teff_method.hfs import (
    HFSLevel,
    allowed_f_values,
    fit_hfs_profile,
    generate_hfs_components,
    hfs_absorption,
)

TRANSITIONS = tuple(
    (spin, lower_j, upper_j)
    for spin, lower_j, upper_j in product(
        (0, 0.5, 1, 1.5, 3.5),
        (0, 0.5, 1, 1.5, 2, 2.5, 3.5),
        (0, 0.5, 1, 1.5, 2, 2.5, 3.5),
    )
    if abs(lower_j - upper_j) <= 1
    and lower_j + upper_j >= 1
    and int(2 * (lower_j + upper_j)) % 2 == 0
)


@pytest.mark.parametrize("spin,lower_j,upper_j", TRANSITIONS)
def test_component_strength_marginals_centroid_and_transition_reversal(spin, lower_j, upper_j):
    lower = HFSLevel(lower_j, 0.035, -0.006)
    upper = HFSLevel(upper_j, 0.11, 0.008)
    components = generate_hfs_components(spin, lower, upper)
    # Summing all electric-dipole branches must reproduce each level's
    # statistical population. This tests the complete 6-j coupling independently
    # of any particular factorial implementation or frozen component table.
    for f in allowed_f_values(spin, lower_j):
        actual = math.fsum(c.strength for c in components if c.F_lower == f)
        expected = (2 * f + 1) / ((2 * spin + 1) * (2 * lower_j + 1))
        assert actual == pytest.approx(expected, rel=0, abs=2e-15)
    for f in allowed_f_values(spin, upper_j):
        actual = math.fsum(c.strength for c in components if c.F_upper == f)
        expected = (2 * f + 1) / ((2 * spin + 1) * (2 * upper_j + 1))
        assert actual == pytest.approx(expected, rel=0, abs=2e-15)
    assert math.fsum(c.strength * c.offset_cm1 for c in components) == pytest.approx(
        0, rel=0, abs=2e-15,
    )
    reverse = {
        (c.F_upper, c.F_lower): c
        for c in generate_hfs_components(spin, upper, lower)
    }
    assert set(reverse) == {(c.F_lower, c.F_upper) for c in components}
    for component in components:
        other = reverse[component.F_lower, component.F_upper]
        assert other.strength == pytest.approx(component.strength, rel=0, abs=2e-15)
        assert other.offset_cm1 == -component.offset_cm1


def invented_pattern():
    # For I=1/2, J=1/2 -> J=3/2, the three allowed branch weights are
    # 1/4, 1/8 and 5/8. No production Sc I constants are used here.
    components = generate_hfs_components(0.5, HFSLevel(0.5, 0.035), HFSLevel(1.5, 0.11))
    assert {(c.F_lower, c.F_upper): c.strength for c in components} == {
        (0, 1): 0.25, (1, 1): 0.125, (1, 2): 0.625,
    }
    parent, cog = 22000.0, 4500.0
    offsets = np.array([-parent * c.offset_cm1 / (cog + c.offset_cm1) for c in components])
    weights = np.array([c.strength for c in components])
    weights /= weights.sum()
    return components, parent, cog, offsets, weights


def test_full_profile_moments_keep_parent_centre_distinct_from_wavelength_centroid():
    components, parent, cog, offsets, weights = invented_pattern()
    amplitude, centre, sigma = 0.63, parent + 0.17, 0.29
    x = parent + np.linspace(-6, 6, 48001)
    absorbed = 1 - hfs_absorption(
        x, amplitude, centre, sigma, components,
        parent_wavelength=parent, cog_wavenumber_cm1=cog,
    )
    area = trapezoid(absorbed, x)
    mean_offset = float(offsets @ weights)
    observed_mean = trapezoid((x - centre) * absorbed, x) / area
    observed_variance = trapezoid((x - centre - observed_mean) ** 2 * absorbed, x) / area
    expected_variance = sigma ** 2 + float((offsets - mean_offset) ** 2 @ weights)
    assert area == pytest.approx(math.sqrt(2 * math.pi) * amplitude * sigma, abs=2e-12)
    assert observed_mean == pytest.approx(mean_offset, rel=0, abs=2e-12)
    assert observed_variance == pytest.approx(expected_variance, rel=0, abs=2e-12)
    # A zero strength-weighted *wavenumber* centroid does not remain exactly
    # zero after the nonlinear wavelength conversion. Do not recenter the
    # pattern on its wavelength centroid or reinterpret sigma as total width.
    assert math.fsum(c.strength * c.offset_cm1 for c in components) == pytest.approx(
        0, rel=0, abs=1e-16,
    )
    assert mean_offset > 5e-6
    assert expected_variance > 2 * sigma ** 2


def test_solver_covariance_matches_independent_finite_difference_curvature():
    components, parent, cog, offsets, weights = invented_pattern()
    rng = np.random.default_rng(92161)
    x = parent + np.sort(rng.uniform(-3, 3, 640))

    def independent_profile(parameters):
        amplitude, centre, sigma = parameters
        profiles = np.exp(-(x[:, None] - centre - offsets[None, :]) ** 2 / (2 * sigma ** 2))
        return 1 - amplitude * np.sum(weights[None, :] * profiles, axis=1)

    truth = np.array([0.44, parent + 0.09, 0.36])
    y = independent_profile(truth) + rng.normal(0, 0.0012, x.size)
    result = fit_hfs_profile(
        x, y, components, parent_wavelength=parent, cog_wavenumber_cm1=cog,
        initial=(0.4, parent, 0.5),
        bounds=((0, parent - 0.5, 0.05), (1, parent + 0.5, 1)),
    )
    assert result.status == "converged"
    parameters = np.array([result.amplitude, result.center_angstrom, result.sigma_angstrom])
    np.testing.assert_allclose(parameters, truth, rtol=0, atol=0.001)
    columns = []
    for index, step in enumerate((1e-6, 1e-4, 1e-5)):
        perturbation = np.zeros(3)
        perturbation[index] = step
        columns.append((
            independent_profile(parameters + perturbation)
            - independent_profile(parameters - perturbation)
        ) / (2 * step))
    jacobian = np.column_stack(columns)
    residuals = independent_profile(parameters) - y
    expected = np.linalg.inv(jacobian.T @ jacobian) * np.sum(residuals ** 2) / (x.size - 3)
    np.testing.assert_allclose(result.covariance, expected, rtol=1e-6, atol=1e-17)
    gradient = math.sqrt(2 * math.pi) * np.array([parameters[2], 0, parameters[0]])
    propagated = math.sqrt(float(gradient @ expected @ gradient))
    assert result.area_formal_error_angstrom == pytest.approx(propagated, rel=5e-7)
