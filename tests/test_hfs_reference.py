"""Fixed synthetic reference comparisons work without a private repository.

The fixture was produced by importing only the allowlisted standalone math
module identified by its source/blob hashes. It contains scalar results from
new analytical profiles and seeded noise, never observations or spectral arrays.
"""
# ruff: noqa: E402

import hashlib
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sc_teff_method import LINE_IDS
from sc_teff_method.hfs import hfs_absorption
from sc_teff_method.hfs_data import load_hfs_atomic_data
from sc_teff_method.hfs_fitting import HFSParameters, fit_hfs_lines
from sc_teff_method.spectrum import prepare_spectrum

FIXTURE = Path(__file__).parent/"fixtures/hfs_analytical_reference.json"
FIXTURE_SHA256 = "2480975f84e54ba662b78717e0e2503e4f74e284087069bc89a7a99813aa46ad"
RECORDS = json.loads(FIXTURE.read_text())


def test_frozen_scalar_fixture_integrity_and_allowed_synthetic_provenance():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert RECORDS["reference_source"]["sha256"] == (
        "245ca2da19e0d0cbc524e762a5722b56b8e223be972069643b55635a075d16e8"
    )
    assert tuple(record["line_id"] for record in RECORDS["cases"]) == LINE_IDS
    assert RECORDS["atomic_dataset_id"] == "hala2022_ab_generated"
    assert RECORDS["options"]["weighting"] == "unweighted"
    assert all(
        difference == 0 for difference in RECORDS["maximum_public_private_differences"].values()
    )
    assert "no observational" in RECORDS["origin"].lower()
    encoded = FIXTURE.read_text()
    assert '"flux"' not in encoded
    assert '"wavelength_air_angstrom"' not in encoded
    assert "/home/" not in encoded


@pytest.mark.parametrize("record", RECORDS["cases"], ids=LINE_IDS)
def test_public_adapter_matches_frozen_private_math_on_new_analytical_profiles(record):
    atomic = load_hfs_atomic_data()
    line = record["line_id"]
    definition, components = atomic.lines[line], atomic.components[line]
    rest, cog = definition.wavelength_air_angstrom, definition.cog_wavenumber_cm1
    truth = record["truth"]
    amplitude, centre, sigma = (
        truth["amplitude"], rest+truth["centre_offset_angstrom"], truth["sigma_angstrom"],
    )
    x = rest+np.linspace(-4, 4, 321)
    strengths = np.array([component.strength for component in components])
    strengths /= np.sum(strengths)
    profile = np.ones(x.shape)
    for component, strength in zip(components, strengths, strict=True):
        offset = -rest*component.offset_cm1/(cog+component.offset_cm1)
        profile -= amplitude*strength*np.exp(-(x-centre-offset)**2/(2*sigma**2))
    evaluated = hfs_absorption(x, amplitude, centre, sigma, components,
                               parent_wavelength=rest, cog_wavenumber_cm1=cog)
    np.testing.assert_allclose(evaluated, profile, rtol=0, atol=1e-15)
    flux = profile+np.random.default_rng(record["noise_seed"]).normal(
        0, record["noise_sigma"], x.size,
    )
    spectrum = prepare_spectrum(
        x, flux, wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
    )
    initial = record["initial"]
    fit = fit_hfs_lines(spectrum, initial_parameters={line: HFSParameters(
        initial["amplitude"], rest+initial["centre_offset_angstrom"], initial["sigma_angstrom"],
    )})[line]
    expected = record["expected"]
    assert fit.status == "converged"
    assert fit.n_pixels == expected["n_pixels"]
    assert fit.flags == tuple(expected["flags"])
    assert fit.parameters.amplitude == pytest.approx(expected["amplitude"], abs=2e-8)
    assert fit.parameters.centre_air_angstrom == pytest.approx(
        expected["centre_air_angstrom"], abs=2e-8,
    )
    assert fit.parameters.sigma_angstrom == pytest.approx(expected["sigma_angstrom"], abs=2e-8)
    assert fit.area_angstrom == pytest.approx(expected["area_angstrom"], abs=2e-8)
    assert fit.area_error_angstrom == pytest.approx(
        expected["area_error_angstrom"], rel=2e-6, abs=1e-10,
    )
    assert fit.residual_rms == pytest.approx(expected["residual_rms"], abs=1e-10)
    assert fit.area_angstrom == pytest.approx(
        math.sqrt(2*math.pi)*fit.parameters.amplitude*fit.parameters.sigma_angstrom, abs=1e-15,
    )
