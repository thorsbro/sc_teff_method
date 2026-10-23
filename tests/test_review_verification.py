"""Independent review checks use newly constructed Gaussian arrays only."""

import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from sc_teff_method import LINE_IDS, estimate_temperature, get_calibration, subset_performance

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
fitting = importlib.import_module("sc_teff_method.fitting")
spectrum_input = importlib.import_module("sc_teff_method.spectrum")
synthetic = importlib.import_module("sc_teff_method.synthetic")

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}
LOADER_SETTINGS = {
    "wavelength_column": "wavelength", "flux_column": "flux",
    "uncertainty_column": None, "mask_column": None,
    "delimiter": ",", "header": True, "skiprows": 0,
}


def independent_input(*, rv_kms=None, only_line1=False):
    """Invent parameters independently of the package's example generator."""
    segments, truth = [], {}
    for i, (line, reference) in enumerate(get_calibration().lines.items()):
        amplitude = 0.23 + 0.017 * i
        centre = reference.wavelength_air_angstrom + 0.09 * math.cos(i+1)
        sigma = 0.28 + 0.021 * (i % 4)
        truth[line] = (amplitude, centre, sigma)
        if not only_line1 or line == "line1":
            segments.append(reference.wavelength_air_angstrom + np.linspace(-4, 4, 321))
    rest_grid = np.concatenate(segments)
    flux = np.ones(rest_grid.shape)
    for amplitude, centre, sigma in truth.values():
        flux -= amplitude * np.exp(-(rest_grid-centre)**2 / (2*sigma**2))
    factor = (1.0 if rv_kms is None else
              math.sqrt((1+rv_kms/299792.458)/(1-rv_kms/299792.458)))
    raw_grid = rest_grid*factor
    spectrum = spectrum_input.prepare_spectrum(raw_grid, flux, rv_kms=rv_kms, **CONVENTIONS)
    checksum = hashlib.sha256(raw_grid.tobytes()+flux.tobytes()).hexdigest()
    identity = {"filename": "analytic.csv", "sha256": checksum,
                "loader_settings": LOADER_SETTINGS}
    return spectrum, truth, identity


def new_session(*, configuration="nandakumar_feh", rv_kms=None, only_line1=False, options=None):
    from sc_teff_method.review import ReviewSession

    spectrum, truth, identity = independent_input(rv_kms=rv_kms, only_line1=only_line1)
    session = ReviewSession.create(
        spectrum, input_identity=identity, configuration=configuration,
        feh=-0.2 if configuration == "nandakumar_feh" else None, options=options,
    )
    return session, truth, identity


def test_review_demo_is_separate_deterministic_and_contains_inspection_cases():
    first = synthetic.make_review_demo()
    second = synthetic.make_review_demo()
    assert synthetic.SYNTHETIC_CASES == (
        "clean", "noise", "missing", "blends", "rv_offset", "variable",
    )
    assert first.seed == 20261023
    np.testing.assert_array_equal(first.flux, second.flux)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.mask.any()
    assert not first.flux.flags.writeable
    assert not first.wavelength_air_angstrom.flags.writeable
    prepared = spectrum_input.prepare_spectrum(
        first.wavelength_air_angstrom, first.flux, uncertainty=first.uncertainty,
        mask=first.mask, **CONVENTIONS,
    )
    fits = fitting.fit_gaussian_lines(prepared)
    for line in ("line1", "line5", "line7", "line9"):
        assert fits[line].status == "converged" and not fits[line].flags
    assert "masked_gap" in fits["line2"].flags
    assert "partial_window_coverage" in fits["line3"].flags
    assert "high_residual_rms" in fits["line4"].flags
    assert "high_residual_rms" in fits["line6"].flags


@pytest.mark.parametrize("configuration", ["nandakumar_feh", "nandakumar_no_feh"])
def test_explicit_accepted_subset_matches_independent_areas_and_existing_api(configuration):
    session, truth, _ = new_session(configuration=configuration)
    assert all(value.state == "pending" for value in session.decisions.values())
    assert session.reviewed.temperature is None
    assert session.reviewed.performance is None
    for line in ("line1", "line4", "line7"):
        session.decide(line, "accepted", note="Inspected an analytical Gaussian.")
    session.decide("line2", "rejected", note="Explicit subset decision in a synthetic test.")
    expected_areas = {
        line: truth[line][0] * truth[line][2] * math.sqrt(2*math.pi)
        for line in ("line1", "line4", "line7")
    }
    expected = estimate_temperature(
        expected_areas, configuration=configuration,
        feh=-0.2 if configuration == "nandakumar_feh" else None,
    )
    final = session.reviewed
    assert final.included_line_ids == ("line1", "line4", "line7")
    assert final.temperature.temperature_k == pytest.approx(expected.temperature_k, abs=1e-7)
    assert final.temperature.line_scatter_k == pytest.approx(expected.line_scatter_k, abs=1e-7)
    assert final.performance == subset_performance(
        final.included_line_ids, configuration=configuration,
    )
    assert any("selection" in warning.lower() and "validat" in warning.lower()
               for warning in final.warnings)


def test_single_line_edits_reuse_pr2_optimizer_and_preserve_other_eight(monkeypatch):
    session, _, _ = new_session()
    session.decide("line1", "accepted")
    session.decide("line4", "accepted")
    original_fits = dict(session.fits)
    original_optimizer = fitting.least_squares
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original_optimizer(*args, **kwargs)

    monkeypatch.setattr(fitting, "least_squares", counted)
    rest = get_calibration().lines["line4"].wavelength_air_angstrom
    window = (rest-2.7, rest+2.7)
    initial = fitting.GaussianParameters(0.3, rest, 0.32)
    regions = ((rest+1.4, rest+1.6),)
    session.edit_line("line4", window=window, initial=initial, excluded_regions=regions)
    assert len(calls) == 1
    for line in LINE_IDS:
        if line != "line4":
            assert session.fits[line] is original_fits[line]
    assert session.decisions["line1"].state == "accepted"
    assert session.decisions["line4"].state == "pending"
    direct = fitting.fit_gaussian_lines(
        session.spectrum, windows={"line4": window}, initial_parameters={"line4": initial},
        excluded_regions={"line4": regions},
        excluded_lines=[line for line in LINE_IDS if line != "line4"],
    )["line4"]
    assert session.fits["line4"] == direct
    session.decide("line4", "accepted")
    session.reset_line("line4")
    assert session.decisions["line4"].state == "pending"
    assert session.controls["line4"].window is None
    assert session.controls["line4"].initial is None
    assert not session.controls["line4"].excluded_regions
    assert session.fits["line4"].window_provisional is True
    assert session.decisions["line1"].state == "accepted"


@pytest.mark.parametrize("rv", [-65.0, 65.0])
def test_saved_review_restores_the_same_accepted_results_without_second_rv(tmp_path, rv):
    from sc_teff_method.review import ReviewSession

    session, truth, identity = new_session(rv_kms=rv)
    for line in ("line1", "line2", "line9"):
        session.decide(line, "accepted", note="Analytical profile inspected.")
    session.decide("line8", "rejected", note="Synthetic subset choice.")
    path = tmp_path / "review.json"
    session.save(path)
    restored = ReviewSession.load(session.spectrum, path, input_identity=identity)
    assert restored.reviewed.temperature == session.reviewed.temperature
    assert restored.reviewed.performance == session.reviewed.performance
    assert restored.spectrum.rv_kms == rv
    assert restored.spectrum.doppler_factor == session.spectrum.doppler_factor
    for line in restored.reviewed.included_line_ids:
        assert restored.fits[line].parameters.centre_air_angstrom == pytest.approx(
            truth[line][1], abs=1e-8,
        )
    payload = json.loads(path.read_text())
    assert not {"flux", "uncertainty", "mask", "wavelength_air_angstrom"} & set(payload["input"])
    assert payload["input"]["filename"] == "analytic.csv"
    stale = {**identity, "sha256": "f"*64}
    with pytest.raises(ValueError):
        ReviewSession.load(session.spectrum, path, input_identity=stale)


def test_failed_and_unavailable_lines_cannot_be_scientifically_accepted():
    failed, _, _ = new_session(options=fitting.FitOptions(max_nfev=1))
    assert failed.fits["line1"].status == "failed"
    with pytest.raises(ValueError):
        failed.decide("line1", "accepted")
    assert failed.reviewed.temperature is None
    partial, _, _ = new_session(only_line1=True)
    assert partial.fits["line2"].status == "unavailable"
    with pytest.raises(ValueError):
        partial.decide("line2", "accepted")
    assert partial.reviewed.temperature is None


@pytest.mark.parametrize("change", ["initial", "same_pixel_mask"])
def test_saved_approval_is_bound_to_controls_even_when_fit_result_is_unchanged(change):
    from sc_teff_method.review import ReviewSession

    session, _, identity = new_session()
    rest = get_calibration().lines["line1"].wavelength_air_angstrom
    if change == "same_pixel_mask":
        session.edit_line("line1", excluded_regions=[(rest+1.131, rest+1.169)])
    session.decide("line1", "accepted", note="Inspected this exact numerical configuration.")
    original = session.fits["line1"]
    payload = session.to_dict()
    controls = payload["lines"]["line1"]["controls"]
    if change == "initial":
        controls["initial"] = {
            "amplitude": 0.3, "centre_air_angstrom": rest+0.04, "sigma_angstrom": 0.31,
        }
        direct = fitting.fit_gaussian_lines(
            session.spectrum,
            initial_parameters={"line1": fitting.GaussianParameters(**controls["initial"])},
            excluded_lines=LINE_IDS[1:],
        )["line1"]
    else:
        controls["excluded_regions"] = [[rest+1.132, rest+1.168]]
        direct = fitting.fit_gaussian_lines(
            session.spectrum, excluded_regions={"line1": controls["excluded_regions"]},
            excluded_lines=LINE_IDS[1:],
        )["line1"]
        assert direct.n_excluded == original.n_excluded
    assert direct.area_angstrom == pytest.approx(original.area_angstrom, rel=0, abs=1e-10)
    with pytest.raises(ValueError, match="[Cc]ontrol|approval"):
        ReviewSession.from_dict(session.spectrum, payload, input_identity=identity)


def test_masking_invented_blend_changes_area_and_same_subset_temperature():
    from sc_teff_method.review import ReviewSession

    example = synthetic.make_review_demo()
    spectrum = spectrum_input.prepare_spectrum(
        example.wavelength_air_angstrom, example.flux, uncertainty=example.uncertainty,
        mask=example.mask, **CONVENTIONS,
    )
    identity = {
        "filename": "demo.csv", "sha256": hashlib.sha256(example.flux.tobytes()).hexdigest(),
        "loader_settings": LOADER_SETTINGS,
    }
    session = ReviewSession.create(spectrum, input_identity=identity, feh=-0.2)
    assert session.fits["line4"].flags
    assert session.decisions["line4"].state == "pending"
    assert session.reviewed.temperature is None
    for line in ("line1", "line4"):
        session.decide(line, "accepted", note="Explicit synthetic comparison.")
    before_area = session.fits["line4"].area_angstrom
    before = session.reviewed.temperature.temperature_k
    rest = get_calibration().lines["line4"].wavelength_air_angstrom
    session.edit_line("line4", excluded_regions=[(rest+0.15, rest+1.60)])
    assert session.decisions["line4"].state == "pending"
    assert session.reviewed.included_line_ids == ("line1",)
    assert session.fits["line4"].flags  # masking does not erase numerical warnings
    session.decide("line4", "accepted",
                   note="Inspected the fitted analytical profile after masking.")
    after_area = session.fits["line4"].area_angstrom
    after = session.reviewed.temperature.temperature_k
    assert abs(after_area-example.truth["line4"].area_angstrom) < (
        abs(before_area-example.truth["line4"].area_angstrom) / 20
    )
    slope = get_calibration().lines["line4"].a_k_per_angstrom
    expected_change = slope * (after_area-before_area) / 2
    assert after-before == pytest.approx(expected_change, rel=0, abs=1e-9)
    assert after > before


def test_scripted_analytical_example_is_portable_reproducible_and_refuses_overwrite(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/demo_interactive_review.py"
    spec = importlib.util.spec_from_file_location("_review_demo_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    output = tmp_path / "demo"
    spectrum_path = example.write_demo(output, scripted_review=True)
    assert spectrum_path.name == "spectrum.csv"
    payload = json.loads((output / "truth.json").read_text())
    comparison = payload["scripted_comparison"]
    assert comparison["acceptance_invalidated_after_mask"] is True
    assert comparison["line4_area_after_angstrom"] < comparison["line4_area_before_angstrom"]
    assert comparison["combined_temperature_after_k"] > comparison["combined_temperature_before_k"]
    saved = json.loads((output / "review.json").read_text())
    assert saved["lines"]["line4"]["decision"]["state"] == "accepted"
    assert saved["lines"]["line6"]["decision"]["state"] == "rejected"
    with pytest.raises(FileExistsError):
        example.write_demo(output, scripted_review=True)
