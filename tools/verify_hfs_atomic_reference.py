"""Compare standalone HFS reference physics using new analytical profiles only.

The sole executable private file is the allowlisted src/scteff/hfs.py, verified
against its frozen checksum before standalone import. No private package is
imported. Only two explicitly allowed atomic CSVs are additionally read. No
spectrum, stellar measurement, fitting catalogue or private history is accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from sc_teff_method import hfs
from sc_teff_method.hfs_data import load_hfs_atomic_data


def verify(reference_root: Path) -> dict:
    atomic = load_hfs_atomic_data()
    reference_files = {}
    for name in ("src/scteff/hfs.py", "config/hfs_components_hala2022_ab.csv",
                 "config/hfs_levels.csv"):
        content = (reference_root/name).read_bytes()
        if hashlib.sha256(content).hexdigest() != atomic.source_hashes[name]:
            raise ValueError(f"Reference checksum differs from frozen numerical source: {name}")
        reference_files[name] = content
    name = "sc_teff_allowed_hfs_reference"
    reference = ModuleType(name)
    sys.modules[name] = reference
    # Execute the bytes already read/verified above, bypassing import loaders:
    # no private __init__, bytecode cache read/write, or additional source read.
    exec(compile(reference_files["src/scteff/hfs.py"], "<allowed HFS reference>", "exec"),
         reference.__dict__)
    rows = list(csv.DictReader(io.StringIO(
        reference_files["config/hfs_components_hala2022_ab.csv"].decode(),
    )))
    levels = [row for row in csv.DictReader(io.StringIO(
        reference_files["config/hfs_levels.csv"].decode(),
    )) if row["dataset_id"] == "hala2022_primary"]
    errors = {name: 0.0 for name in (
        "level_shift_cm1", "generated_offset_cm1", "generated_strength",
        "production_strength", "wavelength_offset_angstrom", "profile_flux",
        "fit_amplitude", "fit_center_angstrom", "fit_sigma_angstrom",
        "fit_area_angstrom", "fit_area_error_angstrom",
    )}
    for row in levels:
        values = float(row["J"]), float(row["A_model_value_cm1"]), float(row["B_model_value_cm1"])
        for f in hfs.allowed_f_values(3.5, values[0]):
            errors["level_shift_cm1"] = max(errors["level_shift_cm1"], abs(
                hfs.hyperfine_energy_shift(3.5, values[0], f, *values[1:])
                - reference.hyperfine_energy_shift(3.5, values[0], f, *values[1:]),
            ))
    rng = np.random.default_rng(20261005)
    for index, (key, line) in enumerate(atomic.lines.items()):
        selected = [row for row in rows if row["line_id"] == key]
        weights = reference._normalize_strengths([float(row["strength"]) for row in selected])
        weights[-1] = 1.0-sum(float(value) for value in weights[:-1])
        reference_components = tuple(
            reference.HFSComponent(float(row["F_lower"]), float(row["F_upper"]),
                                   float(row["component_offset_cm1"]), float(weight))
            for row, weight in zip(selected, weights, strict=True)
        )
        actual = atomic.components[key]
        errors["production_strength"] = max(errors["production_strength"], max(
            abs(left.strength-right.strength)
            for left, right in zip(actual, reference_components, strict=True)
        ))
        lower, upper = (atomic.levels[name].level for name in
                        (line.lower_level_id, line.upper_level_id))
        private_generated = reference.generate_hfs_components(
            3.5, reference.HFSLevel(lower.J, lower.A, lower.B),
            reference.HFSLevel(upper.J, upper.A, upper.B),
        )
        public_generated = hfs.generate_hfs_components(3.5, lower, upper)
        for left, right in zip(public_generated, private_generated, strict=True):
            for field, metric in (("offset_cm1", "generated_offset_cm1"),
                                  ("strength", "generated_strength")):
                errors[metric] = max(
                    errors[metric], abs(getattr(left, field)-getattr(right, field)),
                )
        kwargs = dict(parent_wavelength=line.wavelength_air_angstrom,
                      cog_wavenumber_cm1=line.cog_wavenumber_cm1)
        offsets = hfs.wavenumber_offset_to_wavelength_offset(
            [c.offset_cm1 for c in actual], **kwargs,
        )
        expected_offsets = reference.wavenumber_offset_to_wavelength_offset(
            [c.offset_cm1 for c in reference_components], **kwargs,
        )
        errors["wavelength_offset_angstrom"] = max(errors["wavelength_offset_angstrom"],
                                                    float(np.max(abs(offsets-expected_offsets))))
        parent = line.wavelength_air_angstrom
        x = parent + np.linspace(-3, 3, 601)
        amplitude = 0.48+0.01*index
        centre, sigma = parent+0.07*math.cos(index), 0.32+0.02*(index % 3)
        # Invented truth from the analytical mixture, independent of either evaluator.
        y = 1-amplitude*np.sum(
            np.exp(-0.5*((x[:, None]-centre-offsets)/sigma)**2)
            * np.array([c.strength for c in actual]), axis=1,
        )
        errors["profile_flux"] = max(errors["profile_flux"], float(np.max(abs(
            hfs.hfs_absorption(x, amplitude, centre, sigma, actual, **kwargs)
            - reference.hfs_absorption(x, amplitude, centre, sigma, reference_components, **kwargs),
        ))))
        y = y+rng.normal(0, 0.0015, len(x))
        fit_kwargs = dict(**kwargs, initial=(0.52, parent+0.05, 0.36),
                          bounds=((0, parent-1, 0.03), (1, parent+1, 1.5)))
        expected = reference.fit_hfs_profile(x, y, reference_components, **fit_kwargs)
        fitted = hfs.fit_hfs_profile(x, y, actual, **fit_kwargs)
        assert expected.status == fitted.status == "converged"
        for field, metric in (("amplitude", "fit_amplitude"),
                              ("center_angstrom", "fit_center_angstrom"),
                              ("sigma_angstrom", "fit_sigma_angstrom"),
                              ("area_angstrom", "fit_area_angstrom"),
                              ("area_formal_error_angstrom", "fit_area_error_angstrom")):
            errors[metric] = max(
                errors[metric], abs(getattr(fitted, field)-getattr(expected, field)),
            )
    return {"synthetic_seed": 20261005, "component_count": 133,
            "maximum_absolute_differences": errors,
            "source_sha256": {name: atomic.source_hashes[name] for name in reference_files},
            "inputs": "Only new analytical Gaussian mixtures and independently generated noise."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.reference_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
