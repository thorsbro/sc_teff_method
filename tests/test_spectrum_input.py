"""Input tests use only artificial numeric arrays and analytically generated profiles."""
# ruff: noqa: E402

import math
from dataclasses import FrozenInstanceError

import pytest

np = pytest.importorskip("numpy")

from sc_teff_method.spectrum import SPEED_OF_LIGHT_KMS, Spectrum, load_spectrum, prepare_spectrum

CONVENTIONS = {
    "wavelength_unit": "angstrom", "wavelength_frame": "air", "continuum_normalised": True,
}


def profile():
    wavelength = np.linspace(21727, 21734, 141)
    flux = 1 - 0.2 * np.exp(-0.5 * ((wavelength - 21730.467) / 0.4)**2)
    return wavelength, flux


def test_prepare_copies_read_only_arrays_and_keeps_all_pixels():
    wavelength, flux = profile()
    uncertainty = np.full(wavelength.size, 0.01)
    mask = np.zeros(wavelength.size, dtype=bool)
    mask[10:14] = True
    spectrum = prepare_spectrum(
        wavelength, flux, uncertainty=uncertainty, mask=mask, **CONVENTIONS,
    )
    original = flux.copy()
    flux[:] = 0
    uncertainty[:] = 1
    wavelength[:] += 10
    mask[:] = False
    np.testing.assert_array_equal(spectrum.flux, original)
    np.testing.assert_array_equal(spectrum.uncertainty, np.full(spectrum.n_pixels, 0.01))
    assert spectrum.mask.sum() == 4
    assert spectrum.n_pixels == 141
    assert spectrum.rv_kms is None
    assert spectrum.doppler_factor == 1
    for values in (
        spectrum.wavelength_air_angstrom, spectrum.flux, spectrum.uncertainty, spectrum.mask,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 0
    with pytest.raises(FrozenInstanceError):
        spectrum.rv_kms = 0


@pytest.mark.parametrize("velocity", [-120.0, 0.0, 65.0])
def test_relativistic_rv_sign_and_exact_rest_frame_recovery(velocity):
    rest, flux = profile()
    beta = velocity / SPEED_OF_LIGHT_KMS
    factor = math.sqrt((1 + beta) / (1 - beta))
    shifted = rest * factor
    result = prepare_spectrum(shifted, flux, rv_kms=velocity, **CONVENTIONS)
    np.testing.assert_allclose(result.wavelength_air_angstrom, rest, rtol=0, atol=4e-12)
    np.testing.assert_array_equal(result.flux, flux)
    assert result.rv_kms == velocity
    assert result.doppler_factor == factor
    if velocity > 0:
        assert np.all(result.wavelength_air_angstrom < shifted)
    elif velocity < 0:
        assert np.all(result.wavelength_air_angstrom > shifted)


def test_declaring_zero_velocity_does_not_implicitly_correct_a_shift():
    wavelength, flux = profile()
    shifted = wavelength * 1.001
    result = prepare_spectrum(shifted, flux, rv_kms=0, **CONVENTIONS)
    np.testing.assert_array_equal(result.wavelength_air_angstrom, shifted)
    assert result.rv_kms == 0


@pytest.mark.parametrize("key,value", [
    ("wavelength_unit", "nm"), ("wavelength_unit", "auto"),
    ("wavelength_frame", "vacuum"), ("wavelength_frame", "auto"),
    ("continuum_normalised", False), ("continuum_normalised", 1),
])
def test_conventions_must_be_explicit_and_supported(key, value):
    declarations = dict(CONVENTIONS)
    declarations[key] = value
    with pytest.raises(ValueError, match=key):
        prepare_spectrum(*profile(), **declarations)


def test_omitting_convention_acknowledgements_is_an_error():
    with pytest.raises(TypeError):
        prepare_spectrum(*profile())


@pytest.mark.parametrize("wavelength", [
    [], [1], [2, 1], [1, 1], [1, float("nan")], [1, float("inf")], [0, 1], [-1, 1],
])
def test_invalid_wavelength_grid(wavelength):
    with pytest.raises(ValueError):
        prepare_spectrum(wavelength, np.ones(len(wavelength)), **CONVENTIONS)


@pytest.mark.parametrize("name,value", [
    ("wavelength", [[1, 2], [3, 4]]), ("flux", [[1, 1], [1, 1]]),
    ("uncertainty", [[1, 1], [1, 1]]), ("mask", [[False, False], [False, False]]),
])
def test_non_vector_inputs(name, value):
    kwargs = {"wavelength": [1, 2], "flux": [1, 1], **CONVENTIONS}
    kwargs[name] = value
    with pytest.raises(ValueError, match="one-dimensional"):
        prepare_spectrum(**kwargs)


@pytest.mark.parametrize("name,value", [
    ("flux", [1]), ("uncertainty", [0.01]), ("mask", [False]),
])
def test_pixel_array_lengths_must_match(name, value):
    kwargs = {"wavelength": [1, 2], "flux": [1, 1], **CONVENTIONS}
    kwargs[name] = value
    with pytest.raises(ValueError, match="same length"):
        prepare_spectrum(**kwargs)


@pytest.mark.parametrize("name", ["wavelength", "flux", "uncertainty"])
@pytest.mark.parametrize("value", [[True, True], [1.0, False], [1+0j, 2+0j], ["1", "2"]])
def test_numeric_arrays_reject_boolean_complex_and_string_values(name, value):
    kwargs = {"wavelength": [1, 2], "flux": [1, 1], **CONVENTIONS}
    kwargs[name] = value
    with pytest.raises(TypeError, match="real numbers"):
        prepare_spectrum(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_flux_requires_an_explicit_mask(value):
    with pytest.raises(ValueError, match="finite"):
        prepare_spectrum([1, 2, 3], [1, value, 1], **CONVENTIONS)
    masked = prepare_spectrum(
        [1, 2, 3], [1, value, 1], mask=[False, True, False], **CONVENTIONS,
    )
    assert masked.mask[1]
    assert not np.isfinite(masked.flux[1])


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_uncertainty_requires_an_explicit_mask(value):
    with pytest.raises(ValueError, match="positive and finite"):
        prepare_spectrum([1, 2, 3], [1, 1, 1], uncertainty=[0.01, value, 0.01], **CONVENTIONS)
    masked = prepare_spectrum(
        [1, 2, 3], [1, 1, 1], uncertainty=[0.01, value, 0.01],
        mask=[False, True, False], **CONVENTIONS,
    )
    assert masked.mask[1]


def test_fully_masked_spectrum_preserves_unavailable_measurements_for_fitting():
    spectrum = prepare_spectrum(
        [1, 2], [float("nan"), float("nan")], mask=[True, True], **CONVENTIONS,
    )
    assert spectrum.n_pixels == 2
    assert spectrum.mask.all()


def test_masks_are_explicit_boolean_arrays():
    with pytest.raises(TypeError, match="booleans"):
        prepare_spectrum([1, 2], [1, 1], mask=[0, 1], **CONVENTIONS)


@pytest.mark.parametrize("value", [True, "10", 1+0j, []])
def test_invalid_rv_types(value):
    with pytest.raises(TypeError, match="rv_kms"):
        prepare_spectrum(*profile(), rv_kms=value, **CONVENTIONS)


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), SPEED_OF_LIGHT_KMS, -SPEED_OF_LIGHT_KMS, 10**1000,
])
def test_invalid_rv_values(value):
    with pytest.raises(ValueError, match="rv_kms"):
        prepare_spectrum(*profile(), rv_kms=value, **CONVENTIONS)


def test_direct_canonical_record_validates_arrays_and_correction_metadata():
    with pytest.raises(ValueError, match="strictly increasing"):
        Spectrum(np.array([2, 1]), np.ones(2))
    with pytest.raises(ValueError, match="doppler_factor"):
        Spectrum(np.array([1, 2]), np.ones(2), rv_kms=100, doppler_factor=1)


def test_whitespace_ascii_column_selection_and_physical_line_skipping(tmp_path):
    path = tmp_path / "artificial.txt"
    path.write_text("skip this metadata\n# artificial data only\n9 21730.0 1.0\n8 21730.1 0.9\n")
    spectrum = load_spectrum(
        path, wavelength_column=1, flux_column=2, skiprows=1, **CONVENTIONS,
    )
    np.testing.assert_array_equal(spectrum.wavelength_air_angstrom, [21730, 21730.1])
    np.testing.assert_array_equal(spectrum.flux, [1, 0.9])


def test_csv_named_columns_uncertainties_and_masks(tmp_path):
    path = tmp_path / "artificial.csv"
    path.write_text(
        '# analytical numbers\n"wave",flux,error,exclude\n'
        '21730.0,1.0,0.01,0\n21730.1,nan,nan,1\n21730.2,0.8,0.02,0\n'
    )
    spectrum = load_spectrum(
        path, wavelength_column="wave", flux_column="flux", uncertainty_column="error",
        mask_column="exclude", delimiter=",", header=True, **CONVENTIONS,
    )
    np.testing.assert_array_equal(spectrum.mask, [False, True, False])
    assert np.isnan(spectrum.flux[1])
    np.testing.assert_array_equal(spectrum.uncertainty[[0, 2]], [0.01, 0.02])


def test_declared_semicolon_csv_and_header_indices(tmp_path):
    path = tmp_path / "artificial.txt"
    path.write_text("flux;wavelength;mask\n1;21730;0.0\n0.8;21730.1;1.0\n")
    spectrum = load_spectrum(
        path, wavelength_column=1, flux_column=0, mask_column=2,
        delimiter=";", header=True, **CONVENTIONS,
    )
    assert spectrum.mask.tolist() == [False, True]
    assert spectrum.flux.tolist() == [1, 0.8]


def test_csv_rv_correction_matches_array_interface(tmp_path):
    path = tmp_path / "artificial.csv"
    rest, flux = profile()
    velocity = 65
    factor = math.sqrt((1 + velocity / SPEED_OF_LIGHT_KMS) / (1 - velocity / SPEED_OF_LIGHT_KMS))
    np.savetxt(path, np.column_stack([rest * factor, flux]), delimiter=",", fmt="%.17g")
    spectrum = load_spectrum(path, delimiter=",", rv_kms=velocity, **CONVENTIONS)
    np.testing.assert_allclose(spectrum.wavelength_air_angstrom, rest, rtol=0, atol=4e-12)
    np.testing.assert_array_equal(spectrum.flux, flux)


@pytest.mark.parametrize("content,match", [
    ("", "no data"), ("# only comments\n", "no data"),
    ("21730 1\n\n21730.1 1\n", "Empty"),
    ("21730 1\n21730.1 1 0.1\n", "Expected 2 columns"),
    ("wave flux\n21730 1\n", "Invalid numeric"),
    ("21730 1\n21730.1 bogus\n", "Invalid numeric"),
])
def test_ascii_malformed_and_empty_rows_fail(tmp_path, content, match):
    path = tmp_path / "artificial.txt"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_spectrum(path, **CONVENTIONS)


@pytest.mark.parametrize("content,kwargs,match", [
    ("wave,flux\n21730,\n21730.1,1\n", {}, "Missing flux"),
    ("wave,wave\n21730,1\n", {}, "distinct"),
    ("wave,\n21730,1\n", {}, "non-empty"),
    ("wave,flux\n", {}, "no data"),
    ('wave,flux\n21730,"1\n', {}, "Malformed CSV"),
    ("wave,flux\n21730,1\n21730.1,1\n", {"flux_column": "wrong"}, "Unknown flux"),
    ("wave,flux\n21730,1\n21730.1,1\n", {"flux_column": 2}, "outside"),
    ("wave,flux,mask\n21730,1,0.5\n21730.1,1,0\n", {"mask_column": 2}, "Mask"),
])
def test_csv_malformed_columns_fail(tmp_path, content, kwargs, match):
    path = tmp_path / "artificial.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_spectrum(path, delimiter=",", header=True, **CONVENTIONS, **kwargs)


@pytest.mark.parametrize("kwargs,error", [
    ({"header": 1}, TypeError), ({"skiprows": True}, TypeError),
    ({"skiprows": -1}, ValueError), ({"skiprows": 1.5}, TypeError),
    ({"delimiter": "auto"}, ValueError), ({"delimiter": "#"}, ValueError),
    ({"delimiter": "\n"}, ValueError), ({"wavelength_column": True}, TypeError),
    ({"wavelength_column": -1}, ValueError), ({"wavelength_column": "wave"}, ValueError),
    ({"flux_column": None}, ValueError), ({"uncertainty_column": []}, TypeError),
])
def test_invalid_parser_options_are_rejected_before_reading(tmp_path, kwargs, error):
    path = tmp_path / "artificial.txt"
    path.write_text("21730 1\n21730.1 1\n")
    with pytest.raises(error):
        load_spectrum(path, **CONVENTIONS, **kwargs)


@pytest.mark.parametrize("suffix", [".fits", ".fit", ".fts", ".fits.gz"])
def test_fits_is_explicitly_unsupported_without_reading_the_file(tmp_path, suffix):
    path = tmp_path / ("not_created" + suffix)
    with pytest.raises(ValueError, match="FITS"):
        load_spectrum(path, **CONVENTIONS)


def test_delimiter_is_not_inferred_from_extension(tmp_path):
    path = tmp_path / "artificial.csv"
    path.write_text("21730,1\n21730.1,1\n")
    with pytest.raises(ValueError):
        load_spectrum(path, **CONVENTIONS)
