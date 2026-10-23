"""Local scientific review state and portable, spectrum-free review documents.

This module reuses the PR 2 Gaussian fitter and PR 1 calibration APIs. Numerical
convergence and the user's scientific acceptance are deliberately separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

import numpy as np

from . import __version__
from ._data import CALIBRATION_ID, LINE_IDS, SCOPES, get_calibration
from .api import _finite_real, estimate_temperature, subset_performance
from .continuum import ContinuumCorrection, ContinuumOptions, apply_continuum
from .fitting import (
    FitOptions,
    GaussianParameters,
    LineFit,
    _bounds,
    _interval,
    _regions,
    _start,
    fit_gaussian_lines,
)
from .spectrum import Spectrum, _doppler_factor, _velocity
from .workflow import SpectralEstimate, estimate_from_spectrum

REVIEW_FORMAT = "sc-teff-review"
REVIEW_SCHEMA_VERSION = 2
REVIEW_STATES = ("pending", "accepted", "rejected")
SELECTION_WARNING = (
    "Manual line selection can introduce selection effects. Frozen exact-subset bias and RMSE "
    "do not validate an arbitrary post-hoc line-selection procedure or newly chosen "
    "continuum corrections."
)
_UNSET = object()
_LOADER_DEFAULTS = {
    "wavelength_column": 0,
    "flux_column": 1,
    "uncertainty_column": None,
    "mask_column": None,
    "delimiter": None,
    "header": False,
    "skiprows": 0,
}
_SNAPSHOT_FIELDS = (
    "status",
    "parameters",
    "area_angstrom",
    "area_error_angstrom",
    "parameter_errors",
    "residual_rms",
    "n_pixels",
    "n_excluded",
    "centre_displacement_angstrom",
    "flags",
    "window",
    "window_provisional",
    "reduced_chi2",
    "weighting",
)


@dataclass(frozen=True)
class LineControls:
    """Per-line overrides; None means the existing PR 2 default or auto start."""

    window: tuple[float, float] | None = None
    initial: GaussianParameters | None = None
    excluded_regions: tuple[tuple[float, float], ...] = ()
    numerically_excluded: bool = False
    continuum: ContinuumCorrection = ContinuumCorrection()


@dataclass(frozen=True)
class LineDecision:
    """The user's scientific decision, independent of fit convergence/flags."""

    state: str = "pending"
    note: str = ""
    controls_fingerprint: str | None = None


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, GaussianParameters):
        return asdict(value)
    return value


def _text(value, label: str, *, maximum: int = 10000) -> str:
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} must be a string of at most {maximum} characters without NUL")
    return value


def _keys(value, required: Iterable[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _line_id(value: str) -> str:
    if not isinstance(value, str) or value not in LINE_IDS:
        raise ValueError("Unknown line ID; use line1 through line9")
    return value


def _hash_string(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _loader_settings(value) -> dict:
    if not isinstance(value, Mapping) or set(value) - set(_LOADER_DEFAULTS):
        raise ValueError("loader_settings has unknown fields or is not an object")
    result = {**_LOADER_DEFAULTS, **value}
    if not isinstance(result["header"], bool):
        raise ValueError("loader_settings.header must be boolean")
    if (
        isinstance(result["skiprows"], bool)
        or not isinstance(result["skiprows"], int)
        or result["skiprows"] < 0
    ):
        raise ValueError("loader_settings.skiprows must be a non-negative integer")
    delimiter = result["delimiter"]
    if delimiter is not None and (
        not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in "\r\n#"
    ):
        raise ValueError("Invalid loader_settings delimiter")
    for key in ("wavelength_column", "flux_column", "uncertainty_column", "mask_column"):
        selector = result[key]
        if selector is None and key in ("uncertainty_column", "mask_column"):
            continue
        if isinstance(selector, bool) or not isinstance(selector, (int, str)):
            raise ValueError(f"Invalid loader_settings.{key}")
        if isinstance(selector, int) and selector < 0:
            raise ValueError(f"Invalid loader_settings.{key}")
        if isinstance(selector, str) and (not selector.strip() or not result["header"]):
            raise ValueError(f"Named loader_settings.{key} requires header=True")
    return result


def _identity(value) -> dict:
    _keys(value, ("filename", "sha256", "loader_settings"), "input_identity")
    filename = _text(value["filename"], "filename", maximum=255)
    if not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
        raise ValueError("filename must be a portable basename, not a path")
    return {
        "filename": filename,
        "sha256": _hash_string(value["sha256"], "input checksum"),
        "loader_settings": _loader_settings(value["loader_settings"]),
    }


def _prepared_digest(spectrum: Spectrum) -> str:
    """Hash canonical prepared arrays without embedding any spectrum in JSON."""
    digest = hashlib.sha256(b"sc-teff-prepared-spectrum-v1\n")
    for name in ("wavelength_air_angstrom", "flux", "uncertainty", "mask"):
        digest.update(name.encode() + b"\n")
        values = getattr(spectrum, name)
        if values is None:
            digest.update(b"absent\n")
            continue
        if name == "mask":
            array = np.asarray(values, dtype="u1")
        else:
            array = np.array(values, dtype="<f8", copy=True)
            array[np.isnan(array)] = np.nan  # canonical NaN payload for portability
        digest.update(len(array).to_bytes(8, "little"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _snapshot(fit: LineFit) -> dict:
    result = {name: _plain(getattr(fit, name)) for name in _SNAPSHOT_FIELDS}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**result, "fingerprint": hashlib.sha256(encoded.encode()).hexdigest()}


def _controls_fingerprint(
    controls: LineControls, options: FitOptions, continuum_options: ContinuumOptions,
) -> str:
    payload = {
        "controls": _plain(asdict(controls)), "fit_options": asdict(options),
        "continuum_options": asdict(continuum_options),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _valid_fit(fit: LineFit) -> bool:
    return (
        fit.status == "converged"
        and fit.parameters is not None
        and fit.area_angstrom is not None
        and math.isfinite(fit.area_angstrom)
        and fit.area_angstrom > 0
    )


def _snapshot_matches(saved: dict, current: dict) -> bool:
    """Allow negligible numerical drift, never material changes to an approved fit."""
    exact = ("status", "flags", "n_pixels", "n_excluded", "window_provisional", "weighting")
    if any(saved[key] != current[key] for key in exact):
        return False

    def close(left, right, path=""):
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(close(left[k], right[k], k) for k in left)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))
        if left is None or right is None:
            return left is right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            # Centre coordinates are large; avoid a relative tolerance in Angstrom.
            tolerance = 5e-9 if path == "centre_air_angstrom" else 1e-10
            return math.isclose(left, right, rel_tol=0, abs_tol=tolerance)
        return left == right

    return all(close(saved[key], current[key], key) for key in _SNAPSHOT_FIELDS if key not in exact)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _validate_snapshot(value, line: str) -> None:
    if value["status"] not in ("converged", "failed", "unavailable", "excluded"):
        raise ValueError(f"Invalid saved fit status for {line}")
    if (
        not isinstance(value["flags"], list)
        or len(value["flags"]) > 100
        or any(not isinstance(flag, str) for flag in value["flags"])
    ):
        raise ValueError(f"Invalid saved quality flags for {line}")
    if not isinstance(value["window_provisional"], bool):
        raise ValueError("window_provisional must be boolean")
    _interval(value["window"], "saved fit window", positive=True)
    if value["weighting"] not in ("unweighted", "inverse_sigma"):
        raise ValueError("Unknown saved fitting weighting")
    for key in ("n_pixels", "n_excluded"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise ValueError(f"Invalid saved {key}")
    for key in (
        "area_angstrom",
        "area_error_angstrom",
        "residual_rms",
        "reduced_chi2",
        "centre_displacement_angstrom",
    ):
        if value[key] is not None:
            _finite_real(value[key], f"saved {key}")
    for key in ("parameters", "parameter_errors"):
        if value[key] is not None:
            _keys(value[key], ("amplitude", "centre_air_angstrom", "sigma_angstrom"), key)
            GaussianParameters(**value[key])
    if value["status"] == "converged":
        if (
            value["parameters"] is None
            or value["area_angstrom"] is None
            or value["area_angstrom"] <= 0
        ):
            raise ValueError("A converged saved fit must have positive area and parameters")
    elif value["area_angstrom"] is not None or value["parameters"] is not None:
        raise ValueError("A failed, excluded or unavailable fit must have missing area/parameters")


def _validate_payload(payload) -> dict:
    legacy = isinstance(payload, Mapping) and payload.get("schema_version") == 1
    extra_keys = () if legacy else ("continuum_options",)
    _keys(
        payload,
        (
            "format",
            "schema_version",
            "software",
            "calibration",
            "input",
            "fit_options",
            "lines",
            "notes",
            "scientific_warnings",
            *extra_keys,
        ),
        "review",
    )
    if (
        payload["format"] != REVIEW_FORMAT
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] not in (1, REVIEW_SCHEMA_VERSION)
    ):
        raise ValueError("Unsupported review format or schema version")
    # JSON round-tripping rejects non-finite programmatic fields too.
    try:
        result = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Review fields must be finite JSON values") from exc
    _keys(result["software"], ("name", "version", "numpy", "scipy"), "software")
    if result["software"]["name"] != "sc-teff-method":
        raise ValueError("Unknown review software identity")
    for key, value in result["software"].items():
        _text(value, f"software.{key}", maximum=200)
    calibration = _keys(
        result["calibration"],
        (
            "calibration_id",
            "configuration",
            "feh",
            "reference_temperature_k",
            "validation_scope",
        ),
        "calibration",
    )
    if calibration["calibration_id"] != CALIBRATION_ID:
        raise ValueError("Review uses an unsupported frozen calibration identity")
    specification = get_calibration(calibration["configuration"])
    if specification.requires_feh and calibration["feh"] is None:
        raise ValueError("This review calibration requires explicit [Fe/H]")
    if calibration["feh"] is not None:
        _finite_real(calibration["feh"], "review feh")
    if calibration["reference_temperature_k"] is not None:
        if _finite_real(calibration["reference_temperature_k"], "reference temperature") <= 0:
            raise ValueError("Reference temperature must be positive")
    if calibration["validation_scope"] not in SCOPES:
        raise ValueError("Unsupported validation scope")
    source = _keys(
        result["input"],
        (
            "filename",
            "sha256",
            "loader_settings",
            "wavelength_unit",
            "wavelength_frame",
            "continuum_normalised",
            "rv_kms",
            "doppler_factor",
            "n_pixels",
            "prepared_spectrum_sha256",
        ),
        "input",
    )
    _identity({key: source[key] for key in ("filename", "sha256", "loader_settings")})
    if (
        source["wavelength_unit"] != "angstrom"
        or source["wavelength_frame"] != "air"
        or source["continuum_normalised"] is not True
    ):
        raise ValueError("Review must declare continuum-normalised rest air-Angstrom input")
    _hash_string(source["prepared_spectrum_sha256"], "prepared spectrum checksum")
    velocity = _velocity(source["rv_kms"])
    if not math.isclose(
        _finite_real(source["doppler_factor"], "Doppler factor"),
        _doppler_factor(velocity),
        rel_tol=1e-14,
        abs_tol=0,
    ):
        raise ValueError("Saved Doppler factor does not match its RV correction")
    if (
        isinstance(source["n_pixels"], bool)
        or not isinstance(source["n_pixels"], int)
        or source["n_pixels"] < 2
    ):
        raise ValueError("Invalid saved pixel count")
    _keys(result["fit_options"], asdict(FitOptions()), "fit_options")
    try:
        options = FitOptions(**result["fit_options"])
    except TypeError as exc:
        raise ValueError("Invalid fit_options fields") from exc
    if legacy:
        result["continuum_options"] = asdict(ContinuumOptions())
    _keys(result["continuum_options"], asdict(ContinuumOptions()), "continuum_options")
    continuum_options = ContinuumOptions(**result["continuum_options"])
    _keys(result["lines"], LINE_IDS, "lines")
    for line in LINE_IDS:
        entry = _keys(result["lines"][line], ("controls", "decision", "fit"), line)
        controls = _keys(
            entry["controls"],
            (
                "window",
                "initial",
                "excluded_regions",
                "numerically_excluded",
                *(() if legacy else ("continuum",)),
            ),
            f"{line}.controls",
        )
        if not legacy:
            _keys(controls["continuum"], asdict(ContinuumCorrection()), f"{line}.continuum")
            ContinuumCorrection(**controls["continuum"])
        if not isinstance(controls["numerically_excluded"], bool):
            raise ValueError("numerically_excluded must be boolean")
        rest = specification.lines[line].wavelength_air_angstrom
        window = (
            (rest - options.window_half_width_angstrom, rest + options.window_half_width_angstrom)
            if controls["window"] is None
            else _interval(controls["window"], "window", positive=True)
        )
        reference, lower, upper = _bounds(rest, window, options)
        _regions(controls["excluded_regions"], "excluded_regions")
        if controls["initial"] is not None:
            _keys(
                controls["initial"],
                (
                    "amplitude",
                    "centre_air_angstrom",
                    "sigma_angstrom",
                ),
                f"{line}.initial",
            )
            _start(GaussianParameters(**controls["initial"]), reference, lower, upper)
        decision = _keys(
            entry["decision"],
            ("state", "note", "controls_fingerprint"),
            f"{line}.decision",
        )
        if decision["state"] not in REVIEW_STATES:
            raise ValueError(f"Invalid review decision for {line}")
        _text(decision["note"], f"{line}.note", maximum=4000)
        if decision["state"] == "accepted":
            _hash_string(decision["controls_fingerprint"], f"{line} acceptance fingerprint")
        elif decision["controls_fingerprint"] is not None:
            raise ValueError("Only accepted decisions may carry an acceptance fingerprint")
        if legacy:
            # PR 3 fingerprints exclude continuum controls/options. Verify that
            # historical approval before migrating to the identity continuum.
            if decision["state"] == "accepted":
                encoded = json.dumps(
                    {"controls": controls, "fit_options": result["fit_options"]},
                    sort_keys=True, separators=(",", ":"), allow_nan=False,
                )
                if hashlib.sha256(encoded.encode()).hexdigest() != decision["controls_fingerprint"]:
                    raise ValueError(
                        f"Accepted controls changed for {line}; inspect and approve again"
                    )
            controls["continuum"] = _plain(asdict(ContinuumCorrection()))
            if decision["state"] == "accepted":
                normalised = LineControls(
                    window=tuple(controls["window"]) if controls["window"] is not None else None,
                    initial=(GaussianParameters(**controls["initial"])
                             if controls["initial"] is not None else None),
                    excluded_regions=_regions(controls["excluded_regions"], "excluded_regions"),
                    numerically_excluded=controls["numerically_excluded"],
                )
                decision["controls_fingerprint"] = _controls_fingerprint(
                    normalised, options, continuum_options,
                )
        snapshot = _keys(entry["fit"], (*_SNAPSHOT_FIELDS, "fingerprint"), f"{line}.fit")
        _validate_snapshot(snapshot, line)
        expected = _hash_string(snapshot["fingerprint"], f"{line} fit fingerprint")
        packed = {key: snapshot[key] for key in _SNAPSHOT_FIELDS}
        encoded = json.dumps(packed, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(encoded.encode()).hexdigest() != expected:
            raise ValueError(f"Saved fit fingerprint mismatch for {line}")
    _text(result["notes"], "notes")
    if (
        not isinstance(result["scientific_warnings"], list)
        or len(result["scientific_warnings"]) > 200
    ):
        raise ValueError("scientific_warnings must be a bounded list")
    for warning in result["scientific_warnings"]:
        _text(warning, "scientific warning", maximum=4000)
    result["schema_version"] = REVIEW_SCHEMA_VERSION
    return result


def read_review(path: str | Path) -> dict:
    """Read a versioned review before loading its source, rejecting malformed JSON."""
    path = Path(path).expanduser()
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Review file is unexpectedly large; spectra must not be embedded")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    return _validate_payload(payload)


class ReviewSession:
    """Mutable decisions and controls over immutable PR 2 input and cached fits.

    Use ``create`` for a new review and ``load``/``from_dict`` to restore one.
    Editing/refitting one line never optimises the other eight regions.
    """

    def __setattr__(self, name, value):
        if (
            name
            in {
                "spectrum",
                "configuration",
                "feh",
                "reference_temperature_k",
                "validation_scope",
                "options",
                "continuum_options",
            }
            and name in self.__dict__
        ):
            raise AttributeError(f"{name} is fixed for a review; create a new review to change it")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        spectrum: Spectrum,
        *,
        input_identity: dict,
        configuration: str = "nandakumar_feh",
        feh: float | None = None,
        reference_temperature_k: float | None = None,
        validation_scope: str = "strict_strength_domain",
        windows=None,
        initial_parameters=None,
        excluded_regions=None,
        excluded_lines=(),
        options=None,
        continuum_options: ContinuumOptions | None = None,
    ):
        identity = _identity(input_identity)
        if continuum_options is None:
            continuum_options = ContinuumOptions()
        if not isinstance(continuum_options, ContinuumOptions):
            raise TypeError("continuum_options must be ContinuumOptions")
        baseline = estimate_from_spectrum(
            spectrum,
            configuration=configuration,
            feh=feh,
            reference_temperature_k=reference_temperature_k,
            validation_scope=validation_scope,
            windows=windows,
            initial_parameters=initial_parameters,
            excluded_regions=excluded_regions,
            excluded_lines=excluded_lines,
            options=options,
        )
        self.spectrum = spectrum
        self.configuration = baseline.configuration
        self.feh = baseline.feh
        self.reference_temperature_k = baseline.reference_temperature_k
        self.validation_scope = baseline.validation_scope
        self.options = baseline.fit_options
        self.continuum_options = continuum_options
        self._identity = identity
        self._prepared_sha256 = _prepared_digest(spectrum)
        self._baseline = baseline
        self._fits = dict(baseline.fits)
        self._uncorrected_fits = dict(baseline.fits)
        self._controls = {
            line: LineControls(
                window=fit.window if line in (windows or {}) else None,
                initial=(initial_parameters or {}).get(line),
                excluded_regions=fit.excluded_regions,
                numerically_excluded=fit.status == "excluded",
            )
            for line, fit in self._fits.items()
        }
        self._decisions = {
            line: (
                LineDecision("rejected", "Explicitly excluded when review started")
                if fit.status == "excluded"
                else LineDecision()
            )
            for line, fit in self._fits.items()
        }
        self._notes = ""

    @classmethod
    def create(cls, spectrum: Spectrum, **kwargs) -> ReviewSession:
        return cls(spectrum, **kwargs)

    @property
    def input_identity(self) -> Mapping:
        return MappingProxyType(
            {
                **self._identity,
                "loader_settings": MappingProxyType(self._identity["loader_settings"]),
            }
        )

    @property
    def fits(self) -> Mapping[str, LineFit]:
        return MappingProxyType(self._fits)

    @property
    def uncorrected_fits(self) -> Mapping[str, LineFit]:
        """Unity-continuum comparison using the current window/start/masks."""
        return MappingProxyType(self._uncorrected_fits)

    def corrected_spectrum(self, line_id: str) -> Spectrum:
        """Working input actually used by the selected fit; original stays immutable."""
        line = _line_id(line_id)
        spectrum, _ = apply_continuum(
            self.spectrum, self._controls[line].continuum, window=self._fits[line].window,
            excluded_regions=self._controls[line].excluded_regions, options=self.continuum_options,
        )
        return spectrum

    @property
    def controls(self) -> Mapping[str, LineControls]:
        return MappingProxyType(self._controls)

    @property
    def decisions(self) -> Mapping[str, LineDecision]:
        return MappingProxyType(self._decisions)

    @property
    def notes(self) -> str:
        return self._notes

    @notes.setter
    def notes(self, value: str) -> None:
        self._notes = _text(value, "notes")

    def edit_line(
        self, line_id: str, *, window=_UNSET, initial=_UNSET, excluded_regions=_UNSET,
        continuum=_UNSET,
    ) -> LineFit:
        """Atomically apply controls, refit only this line, and require fresh review."""
        line = _line_id(line_id)
        previous = self._controls[line]
        selected_window = previous.window if window is _UNSET else window
        selected_initial = previous.initial if initial is _UNSET else initial
        selected_regions = (
            previous.excluded_regions if excluded_regions is _UNSET else excluded_regions
        )
        selected_continuum = previous.continuum if continuum is _UNSET else continuum
        if not isinstance(selected_continuum, ContinuumCorrection):
            raise TypeError("continuum must be a ContinuumCorrection")
        rest = get_calibration(self.configuration).lines[line].wavelength_air_angstrom
        effective_window = (
            (rest-self.options.window_half_width_angstrom,
             rest+self.options.window_half_width_angstrom)
            if selected_window is None else selected_window
        )
        working, continuum_flags = apply_continuum(
            self.spectrum, selected_continuum, window=effective_window,
            excluded_regions=selected_regions, options=self.continuum_options,
        )
        arguments = dict(
            windows={line: selected_window} if selected_window is not None else None,
            initial_parameters={line: selected_initial} if selected_initial is not None else None,
            excluded_regions={line: selected_regions},
            excluded_lines=tuple(other for other in LINE_IDS if other != line),
            options=self.options,
        )
        measured = fit_gaussian_lines(working, **arguments)[line]
        uncorrected = (
            measured if working is self.spectrum
            else fit_gaussian_lines(self.spectrum, **arguments)[line]
        )
        if continuum_flags:
            measured = replace(measured, flags=(*measured.flags, *continuum_flags))
        controls = LineControls(
            measured.window if selected_window is not None else None,
            selected_initial,
            measured.excluded_regions,
            continuum=selected_continuum,
        )
        self._controls[line] = controls
        self._fits[line] = measured
        self._uncorrected_fits[line] = uncorrected
        self._decisions[line] = LineDecision()
        return measured

    def refit_line(self, line_id: str) -> LineFit:
        return self.edit_line(line_id)

    def reset_line(self, line_id: str) -> LineFit:
        """Restore the PR 2 default window/auto start; remove added interval masks."""
        return self.edit_line(
            line_id, window=None, initial=None, excluded_regions=(),
            continuum=ContinuumCorrection(),
        )

    def reset_continuum(self, line_id: str) -> LineFit:
        """Restore unity, retaining the selected line's other fitting controls."""
        return self.edit_line(line_id, continuum=ContinuumCorrection())

    def decide(self, line_id: str, state: str, note: str = "") -> None:
        line = _line_id(line_id)
        if state not in REVIEW_STATES:
            raise ValueError(f"state must be one of {REVIEW_STATES}")
        note = _text(note, "decision note", maximum=4000)
        if state == "accepted" and not _valid_fit(self._fits[line]):
            raise ValueError("Only a valid converged positive-area fit can be accepted")
        self._decisions[line] = LineDecision(
            state,
            note,
            _controls_fingerprint(self._controls[line], self.options, self.continuum_options)
            if state == "accepted"
            else None,
        )

    @property
    def reviewed(self) -> SpectralEstimate:
        """Return PR 1 results from only explicitly accepted valid cached fits."""
        valid = {line: fit.area_angstrom for line, fit in self._fits.items() if _valid_fit(fit)}
        accepted = {
            line: valid[line]
            for line in LINE_IDS
            if line in valid and self._decisions[line].state == "accepted"
        }
        kwargs = dict(
            configuration=self.configuration,
            feh=self.feh,
            reference_temperature_k=self.reference_temperature_k,
        )
        diagnostic = estimate_temperature(valid, **kwargs) if valid else None
        temperature = estimate_temperature(accepted, **kwargs) if accepted else None
        performance = (
            subset_performance(
                accepted,
                configuration=self.configuration,
                scope=self.validation_scope,
                cohort="exact_subset",
            )
            if accepted
            else None
        )
        excluded = {
            line: (
                f"{self._decisions[line].state}: {self._decisions[line].note}".rstrip(": ")
                if _valid_fit(fit)
                else f"{fit.status}: {fit.message}"
            )
            for line, fit in self._fits.items()
            if line not in accepted
        }
        warnings = [
            "Only explicitly accepted valid Gaussian measurements contribute to reviewed Teff.",
            "Numerical convergence is not scientific acceptance; formal fit errors exclude "
            "continuum and profile systematics.",
            SELECTION_WARNING,
            "Analytical synthetic tests establish software correctness, "
            "not instrument transfer accuracy.",
        ]
        if any(control.continuum.active for control in self._controls.values()):
            warnings.append(
                "Local continuum corrections are entirely user-controlled and do not reproduce "
                "the calibration paper's continuum procedures. Placement changes Gaussian areas "
                "and can introduce systematic measurement differences; formal Gaussian errors "
                "exclude continuum-placement uncertainty. The user must assess each correction."
            )
        if any(fit.window_provisional for fit in self._fits.values()):
            warnings.append(
                "Default symmetric fitting windows are provisional and may need adjustment."
            )
        if self.spectrum.uncertainty is not None:
            warnings.append(
                "Supplied pixel uncertainties enable weighting; the frozen calibration used "
                "unweighted measurements and weighted transfer is not validated."
            )
        if any(self._fits[line].flags for line in accepted):
            warnings.append(
                "Quality-flagged fits were explicitly manually accepted; flags are retained."
            )
        calibration = get_calibration(self.configuration)
        passes_cut = (
            all(area >= calibration.declared_strength_min_angstrom for area in accepted.values())
            if accepted
            else None
        )
        if passes_cut is False and self.validation_scope == "strict_strength_domain":
            warnings.append(
                "An accepted area fails the declared strength cut; strict-domain validation "
                "statistics do not cover this measurement."
            )
        if not accepted:
            warnings.append(
                "No valid lines are explicitly accepted; reviewed temperature is missing."
            )
        return replace(
            self._baseline,
            fits=MappingProxyType(dict(self._fits)),
            line_temperatures_k=MappingProxyType(
                dict(diagnostic.line_temperatures_k) if diagnostic else {}
            ),
            calibration_warnings=diagnostic.warnings if diagnostic else (),
            included_line_ids=tuple(accepted),
            exclusion_reasons=MappingProxyType(excluded),
            temperature=temperature,
            performance=performance,
            quality_policy="manual_review",
            initial_parameters=MappingProxyType(
                {
                    line: control.initial
                    for line, control in self._controls.items()
                    if control.initial is not None
                }
            ),
            target_passes_declared_strength_cut=passes_cut,
            warnings=tuple(warnings),
            continuum_corrections=(
                MappingProxyType({
                    line: control.continuum for line, control in self._controls.items()
                })
                if any(control.continuum != ContinuumCorrection()
                       for control in self._controls.values()) else None
            ),
        )

    def to_dict(self) -> dict:
        """Return a portable audit document; fitted snapshots never replace recalculation."""
        reviewed = self.reviewed
        return {
            "format": REVIEW_FORMAT,
            "schema_version": REVIEW_SCHEMA_VERSION,
            "software": {
                "name": "sc-teff-method",
                "version": __version__,
                "numpy": version("numpy"),
                "scipy": version("scipy"),
            },
            "calibration": {
                "calibration_id": CALIBRATION_ID,
                "configuration": self.configuration,
                "feh": self.feh,
                "reference_temperature_k": self.reference_temperature_k,
                "validation_scope": self.validation_scope,
            },
            "input": {
                **_plain(self.input_identity),
                "wavelength_unit": "angstrom",
                "wavelength_frame": "air",
                "continuum_normalised": True,
                "rv_kms": self.spectrum.rv_kms,
                "doppler_factor": self.spectrum.doppler_factor,
                "n_pixels": self.spectrum.n_pixels,
                "prepared_spectrum_sha256": self._prepared_sha256,
            },
            "fit_options": asdict(self.options),
            "continuum_options": asdict(self.continuum_options),
            "lines": {
                line: {
                    "controls": _plain(asdict(self._controls[line])),
                    "decision": asdict(self._decisions[line]),
                    "fit": _snapshot(self._fits[line]),
                }
                for line in LINE_IDS
            },
            "notes": self.notes,
            "scientific_warnings": list(reviewed.warnings)
            + [warning.message for warning in reviewed.calibration_warnings],
        }

    @classmethod
    def from_dict(cls, spectrum: Spectrum, payload: dict, *, input_identity: dict) -> ReviewSession:
        if not isinstance(spectrum, Spectrum):
            raise TypeError("spectrum must be an already prepared Spectrum")
        payload = _validate_payload(payload)
        identity = _identity(input_identity)
        source = payload["input"]
        if identity["sha256"] != source["sha256"]:
            raise ValueError("Source spectrum checksum does not match this review")
        if identity["loader_settings"] != _loader_settings(source["loader_settings"]):
            raise ValueError("Input parser settings do not match this review")
        if spectrum.rv_kms != source["rv_kms"] or not math.isclose(
            spectrum.doppler_factor, source["doppler_factor"], rel_tol=1e-14, abs_tol=0
        ):
            raise ValueError("RV correction does not match this review; do not correct twice")
        if (
            spectrum.n_pixels != source["n_pixels"]
            or _prepared_digest(spectrum) != source["prepared_spectrum_sha256"]
        ):
            raise ValueError("Prepared spectrum checksum does not match this review")
        windows, initial, regions, excluded_lines = {}, {}, {}, []
        for line in LINE_IDS:
            entry = payload["lines"][line]
            control = entry["controls"]
            if control["window"] is not None:
                windows[line] = control["window"]
            if control["initial"] is not None:
                initial[line] = GaussianParameters(**control["initial"])
            regions[line] = control["excluded_regions"]
            if control["numerically_excluded"]:
                excluded_lines.append(line)
        calibration = payload["calibration"]
        session = cls.create(
            spectrum,
            input_identity=identity,
            configuration=calibration["configuration"],
            feh=calibration["feh"],
            reference_temperature_k=calibration["reference_temperature_k"],
            validation_scope=calibration["validation_scope"],
            windows=windows,
            initial_parameters=initial,
            excluded_regions=regions,
            excluded_lines=excluded_lines,
            options=FitOptions(**payload["fit_options"]),
            continuum_options=ContinuumOptions(**payload["continuum_options"]),
        )
        for line in LINE_IDS:
            correction = ContinuumCorrection(**payload["lines"][line]["controls"]["continuum"])
            if correction != ContinuumCorrection():
                if session._controls[line].numerically_excluded:
                    # Preserve unavailable/excluded state, while retaining manual
                    # controls for a later deliberate refit.
                    session._controls[line] = replace(session._controls[line], continuum=correction)
                    apply_continuum(
                        spectrum, correction, window=session._fits[line].window,
                        excluded_regions=session._controls[line].excluded_regions,
                        options=session.continuum_options,
                    )
                else:
                    session.edit_line(line, continuum=correction)
            decision = payload["lines"][line]["decision"]
            if decision["state"] == "accepted" and decision["controls_fingerprint"] != (
                _controls_fingerprint(
                    session._controls[line], session.options, session.continuum_options,
                )
            ):
                raise ValueError(f"Accepted controls changed for {line}; inspect and approve again")
            if decision["state"] == "accepted" and not _snapshot_matches(
                payload["lines"][line]["fit"],
                _snapshot(session._fits[line]),
            ):
                raise ValueError(
                    f"Accepted fit changed for {line}; inspect and accept the revised fit again"
                )
            session.decide(line, decision["state"], decision["note"])
        session.notes = payload["notes"]
        return session

    @classmethod
    def load(cls, spectrum: Spectrum, path: str | Path, *, input_identity: dict) -> ReviewSession:
        return cls.from_dict(spectrum, read_review(path), input_identity=input_identity)

    def save(self, path: str | Path, *, overwrite: bool = False, protected_paths=()) -> Path:
        """Save atomically, refusing existing files without explicit overwrite permission."""
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be boolean")
        destination = Path(path).expanduser()
        if destination.is_symlink():
            raise ValueError("Review destination must not be a symbolic link")
        if any(
            destination.resolve() == Path(item).expanduser().resolve() for item in protected_paths
        ):
            raise ValueError(
                "Review destination would replace a protected source/configuration file"
            )
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Review already exists: {destination}")
        if (
            destination.exists()
            and destination.is_file()
            and (hashlib.sha256(destination.read_bytes()).hexdigest() == self._identity["sha256"])
        ):
            raise ValueError("Review destination contains the source spectrum; it is protected")
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".sc-teff-review-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, destination)
            else:
                # Atomic no-clobber publication also protects against a file appearing mid-save.
                os.link(temporary, destination)
                os.unlink(temporary)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination
