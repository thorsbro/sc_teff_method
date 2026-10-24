"""Independent, reproducible scientific approval of HFS line measurements.

HFS and Gaussian decisions have distinct identities. No observed spectra are
stored in review JSON; fits are always regenerated from the original input.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

from ._data import CALIBRATION_ID, LINE_IDS
from .continuum import ContinuumCorrection, ContinuumOptions, apply_continuum
from .fitting import GaussianParameters
from .hfs_calibration import estimate_hfs_temperature, hfs_subset_performance
from .hfs_data import ATOMIC_ID, HFS_DATASET_ID
from .hfs_fitting import HFSFitOptions, fit_hfs_lines
from .hfs_workflow import estimate_hfs_from_spectrum
from .measurements import LineAreas
from .outputs import _plain
from .review import (
    LineControls, LineDecision, _identity, _prepared_digest, _snapshot,
    _snapshot_matches, _valid_fit, _strict_object, _reject_constant, _text,
)
from .spectrum import Spectrum

FORMAT = "sc-teff-hfs-review"
VERSION = 1
STATES = ("pending", "accepted", "rejected")
UNSET = object()


def _fingerprint(controls, options, continuum_options):
    data = {
        "controls": _plain(controls),
        "fit_options": asdict(options),
        "continuum_options": asdict(continuum_options),
        "measurement_source": "hfs",
        "atomic_dataset_id": HFS_DATASET_ID,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_hfs_review(path):
    """Read and minimally validate the separate versioned HFS review format."""
    path = Path(path).expanduser()
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("HFS review is too large; spectral arrays must not be embedded")
    data = json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=_strict_object, parse_constant=_reject_constant)
    needed = {"format", "schema_version", "measurement_source", "atomic_dataset_id",
              "calibration", "input", "fit_options", "continuum_options", "lines", "notes"}
    if not isinstance(data, dict) or set(data) != needed:
        raise ValueError("Invalid HFS review keys")
    if (data["format"] != FORMAT or type(data["schema_version"]) is not int
            or data["schema_version"] != VERSION or data["measurement_source"] != "hfs"
            or data["atomic_dataset_id"] != HFS_DATASET_ID):
        raise ValueError("Unsupported HFS review identity")
    if not isinstance(data["lines"], dict) or set(data["lines"]) != set(LINE_IDS):
        raise ValueError("HFS review must contain exactly nine lines")
    _text(data["notes"], "review notes")
    return data


class HFSReviewSession:
    """Review mutable controls against an immutable, prepared source spectrum.

    The numerical solver and empirical calibration are imported from PR #5,
    never duplicated. A line edit refits only that line and clears its approval.
    """

    measurement_source = "hfs"

    def __init__(
        self, spectrum: Spectrum, *, input_identity, configuration="nandakumar_feh",
        feh=None, reference_temperature_k=None, validation_scope="strict_strength_domain",
        windows=None, initial_parameters=None, excluded_regions=None, excluded_lines=(),
        continuum_corrections=None, options=None, continuum_options=None,
        gaussian_anchor_areas=None,
    ):
        self._identity = _identity(input_identity)
        self.spectrum = spectrum
        self._prepared_sha256 = _prepared_digest(spectrum)
        self._baseline = estimate_hfs_from_spectrum(
            spectrum, configuration=configuration, feh=feh,
            reference_temperature_k=reference_temperature_k,
            validation_scope=validation_scope,
            windows=windows, initial_parameters=initial_parameters,
            excluded_regions=excluded_regions, excluded_lines=excluded_lines,
            continuum_corrections=continuum_corrections,
            options=options, continuum_options=continuum_options,
            gaussian_anchor_areas=gaussian_anchor_areas,
        )
        self.configuration = self._baseline.configuration
        self.feh = self._baseline.feh
        self.reference_temperature_k = self._baseline.reference_temperature_k
        self.validation_scope = self._baseline.validation_scope
        self.options = self._baseline.fit_options
        self.continuum_options = self._baseline.continuum_options
        self.gaussian_anchor_areas = self._baseline.gaussian_anchor_areas
        self._fits = dict(self._baseline.fits)
        self._controls = {
            line: LineControls(
                window=fit.window if line in (windows or {}) else None,
                initial=(initial_parameters or {}).get(line),
                excluded_regions=fit.excluded_regions,
                numerically_excluded=(fit.status == "excluded"),
                continuum=(continuum_corrections or {}).get(line, ContinuumCorrection()),
            )
            for line, fit in self._fits.items()
        }
        self._uncorrected_fits = dict(self._fits)
        if any(control.continuum.active for control in self._controls.values()):
            uncorrected = fit_hfs_lines(
                spectrum, windows=windows, initial_parameters=initial_parameters,
                excluded_regions=excluded_regions, excluded_lines=excluded_lines, options=self.options,
            )
            self._uncorrected_fits.update(uncorrected)
        self._decisions = {
            line: (LineDecision("rejected", "Excluded at review creation")
                   if fit.status == "excluded" else LineDecision())
            for line, fit in self._fits.items()
        }
        self._notes = ""

    @classmethod
    def create(cls, spectrum, **kwargs):
        return cls(spectrum, **kwargs)

    @property
    def input_identity(self):
        return MappingProxyType({**self._identity,
            "loader_settings": MappingProxyType(self._identity["loader_settings"])})

    @property
    def fits(self):
        return MappingProxyType(self._fits)

    @property
    def uncorrected_fits(self):
        return MappingProxyType(self._uncorrected_fits)

    @property
    def controls(self):
        return MappingProxyType(self._controls)

    @property
    def decisions(self):
        return MappingProxyType(self._decisions)

    @property
    def notes(self):
        return self._notes

    @notes.setter
    def notes(self, value):
        self._notes = _text(value, "review notes")

    def corrected_spectrum(self, line_id):
        if line_id not in LINE_IDS:
            raise ValueError("Unknown HFS line")
        working, _ = apply_continuum(
            self.spectrum, self._controls[line_id].continuum,
            window=self._fits[line_id].window,
            excluded_regions=self._controls[line_id].excluded_regions,
            options=self.continuum_options,
        )
        return working

    def edit_line(self, line_id, *, window=UNSET, initial=UNSET,
                  excluded_regions=UNSET, continuum=UNSET):
        if line_id not in LINE_IDS:
            raise ValueError("Unknown HFS line")
        old = self._controls[line_id]
        win = old.window if window is UNSET else window
        guess = old.initial if initial is UNSET else initial
        masks = old.excluded_regions if excluded_regions is UNSET else excluded_regions
        correction = old.continuum if continuum is UNSET else continuum
        arguments = dict(
            windows={line_id: win} if win is not None else {},
            initial_parameters={line_id: guess} if guess is not None else {},
            excluded_regions={line_id: masks},
            excluded_lines=[line for line in LINE_IDS if line != line_id],
            options=self.options, continuum_options=self.continuum_options,
        )
        # Perform both fits before mutating review decisions, including on failure.
        fitted = fit_hfs_lines(self.spectrum, continuum_corrections={line_id: correction},
                               **arguments)[line_id]
        original = (fitted if not correction.active else
                    fit_hfs_lines(self.spectrum, **arguments)[line_id])
        self._fits[line_id] = fitted
        self._uncorrected_fits[line_id] = original
        self._controls[line_id] = LineControls(
            window=fitted.window if win is not None else None,
            initial=guess, excluded_regions=fitted.excluded_regions,
            continuum=correction,
        )
        self._decisions[line_id] = LineDecision()
        return fitted

    def refit_line(self, line_id):
        return self.edit_line(line_id)

    def reset_line(self, line_id):
        return self.edit_line(line_id, window=None, initial=None,
                              excluded_regions=(), continuum=ContinuumCorrection())

    def reset_continuum(self, line_id):
        return self.edit_line(line_id, continuum=ContinuumCorrection())

    def decide(self, line_id, state, note=""):
        if line_id not in LINE_IDS or state not in STATES:
            raise ValueError("Unknown line or HFS review state")
        note = _text(note, "review note", maximum=4000)
        if state == "accepted" and not _valid_fit(self._fits[line_id]):
            raise ValueError("Only converged positive-area HFS fits can be accepted")
        self._decisions[line_id] = LineDecision(
            state, note,
            _fingerprint(self._controls[line_id], self.options, self.continuum_options)
            if state == "accepted" else None,
        )

    @property
    def reviewed(self):
        """Combine only explicitly accepted HFS line temperatures."""
        usable = {line: fit.area_angstrom for line, fit in self._fits.items()
                  if _valid_fit(fit)}
        accepted = {line: usable[line] for line in LINE_IDS
                    if line in usable and self._decisions[line].state == "accepted"}
        kwargs = dict(configuration=self.configuration, feh=self.feh,
                      reference_temperature_k=self.reference_temperature_k,
                      gaussian_anchor_areas=self.gaussian_anchor_areas)
        diagnostic = estimate_hfs_temperature(LineAreas(usable, "hfs"), **kwargs) if usable else None
        final = estimate_hfs_temperature(LineAreas(accepted, "hfs"), **kwargs) if accepted else None
        performance = (hfs_subset_performance(
            accepted, configuration=self.configuration, scope=self.validation_scope,
            cohort="exact_subset") if accepted else None)
        excluded = {
            line: (f"{self._decisions[line].state}: {self._decisions[line].note}"
                   if _valid_fit(fit) else f"{fit.status}: {fit.message}")
            for line, fit in self._fits.items() if line not in accepted
        }
        notes = [
            "Only explicitly approved HFS measurements contribute to reviewed Teff.",
            "Gaussian and HFS reviews are independent; numerical convergence is not approval.",
            "The frozen HFS strict-strength-domain selection used Gaussian-anchor areas.",
            "User-selected continuum and line subsets are not validated by frozen RMSE.",
        ]
        if self.gaussian_anchor_areas is None:
            notes.append("Gaussian anchor areas were not supplied; strict-domain applicability is unknown.")
        if any(c.continuum.active for c in self._controls.values()):
            notes.append("HFS continuum corrections are user controlled; formal errors exclude placement uncertainty.")
        if not accepted:
            notes.append("No HFS fits are accepted; reviewed temperature is missing.")
        return replace(
            self._baseline, fits=MappingProxyType(dict(self._fits)),
            line_temperatures_k=MappingProxyType(
                dict(diagnostic.line_temperatures_k) if diagnostic else {}),
            calibration_warnings=diagnostic.warnings if diagnostic else (),
            included_line_ids=tuple(accepted), exclusion_reasons=MappingProxyType(excluded),
            temperature=final, performance=performance, quality_policy="manual_review",
            initial_parameters=MappingProxyType({
                line: control.initial for line, control in self._controls.items()
                if control.initial is not None}),
            continuum_corrections=MappingProxyType({
                line: control.continuum for line, control in self._controls.items()}),
            strict_strength_domain_applicable=(
                final.strict_strength_domain_applicable if final else None),
            warnings=tuple(notes),
        )

    def to_dict(self):
        return {
            "format": FORMAT, "schema_version": VERSION,
            "measurement_source": "hfs", "atomic_dataset_id": HFS_DATASET_ID,
            "calibration": {
                "calibration_id": CALIBRATION_ID, "configuration": self.configuration,
                "feh": self.feh, "reference_temperature_k": self.reference_temperature_k,
                "validation_scope": self.validation_scope,
                "gaussian_anchor_areas": _plain(self.gaussian_anchor_areas),
            },
            "input": {**_plain(self.input_identity),
                "wavelength_unit": "angstrom", "wavelength_frame": "air",
                "continuum_normalised": True, "rv_kms": self.spectrum.rv_kms,
                "doppler_factor": self.spectrum.doppler_factor,
                "n_pixels": self.spectrum.n_pixels,
                "prepared_spectrum_sha256": self._prepared_sha256,
            },
            "fit_options": asdict(self.options),
            "continuum_options": asdict(self.continuum_options),
            "lines": {
                line: {
                    "controls": _plain(self._controls[line]),
                    "decision": asdict(self._decisions[line]),
                    "fit": _snapshot(self._fits[line]),
                }
                for line in LINE_IDS
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, spectrum, payload, *, input_identity):
        if (not isinstance(payload, dict) or payload.get("format") != FORMAT
                or payload.get("schema_version") != VERSION
                or payload.get("measurement_source") != "hfs"
                or payload.get("atomic_dataset_id") != HFS_DATASET_ID
                or set(payload.get("lines", {})) != set(LINE_IDS)):
            raise ValueError("Not a supported HFS review")
        source = payload["input"]
        identity = _identity(input_identity)
        if (source["sha256"] != identity["sha256"]
                or source["loader_settings"] != identity["loader_settings"]
                or source["prepared_spectrum_sha256"] != _prepared_digest(spectrum)
                or source["rv_kms"] != spectrum.rv_kms
                or not math.isclose(source["doppler_factor"], spectrum.doppler_factor,
                                    rel_tol=1e-14)
                or source["n_pixels"] != spectrum.n_pixels
                or source["wavelength_unit"] != "angstrom"
                or source["wavelength_frame"] != "air"
                or source["continuum_normalised"] is not True):
            raise ValueError("Source/preparation/RV checksum differs from HFS review")
        cal = payload["calibration"]
        if cal["calibration_id"] != CALIBRATION_ID:
            raise ValueError("Frozen HFS calibration changed")
        windows, initials, regions, corrections, excluded = {}, {}, {}, {}, []
        for line in LINE_IDS:
            entry = payload["lines"][line]
            controls = entry["controls"]
            if controls["window"] is not None:
                windows[line] = tuple(controls["window"])
            if controls["initial"] is not None:
                initials[line] = GaussianParameters(**controls["initial"])
            regions[line] = controls["excluded_regions"]
            corrections[line] = ContinuumCorrection(**controls["continuum"])
            if controls["numerically_excluded"]:
                excluded.append(line)
        result = cls.create(
            spectrum, input_identity=identity, configuration=cal["configuration"],
            feh=cal["feh"], reference_temperature_k=cal["reference_temperature_k"],
            validation_scope=cal["validation_scope"],
            gaussian_anchor_areas=cal["gaussian_anchor_areas"],
            windows=windows, initial_parameters=initials, excluded_regions=regions,
            continuum_corrections=corrections, excluded_lines=excluded,
            options=HFSFitOptions(**payload["fit_options"]),
            continuum_options=ContinuumOptions(**payload["continuum_options"]),
        )
        for line in LINE_IDS:
            decision = payload["lines"][line]["decision"]
            if decision["state"] == "accepted":
                if decision["controls_fingerprint"] != _fingerprint(
                    result._controls[line], result.options, result.continuum_options,
                ):
                    raise ValueError(f"Accepted HFS controls differ for {line}")
                if not _snapshot_matches(payload["lines"][line]["fit"],
                                         _snapshot(result._fits[line])):
                    raise ValueError(f"Accepted HFS fit changed for {line}")
            result.decide(line, decision["state"], decision["note"])
        result.notes = payload["notes"]
        return result

    @classmethod
    def load(cls, spectrum, path, *, input_identity):
        return cls.from_dict(spectrum, read_hfs_review(path), input_identity=input_identity)

    def save(self, path, *, overwrite=False, protected_paths=()):
        path = Path(path).expanduser()
        if path.is_symlink():
            raise ValueError("Review destination may not be a symlink")
        if any(path.resolve() == Path(item).expanduser().resolve() for item in protected_paths):
            raise ValueError("Review destination would overwrite an input/configuration file")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == self._identity["sha256"]:
            raise ValueError("Review destination contains the protected source spectrum")
        payload = json.dumps(self.to_dict(), allow_nan=False, indent=2, ensure_ascii=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".sc-teff-hfs-review-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(name, path)
            else:
                os.link(name, path)
                os.unlink(name)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return path
