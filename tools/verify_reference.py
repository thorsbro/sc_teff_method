"""Optional maintainer comparison against the read-only private implementation.

Run in an environment with this public package and the reference implementation's
NumPy dependency installed. The default tests and CI never run this tool or need
private access. Only allowlisted numerical tables and source code are read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

from export_frozen_data import PREFIX, export

from sc_teff_method import (
    CALIBRATION_ID,
    COHORTS,
    CONFIGURATIONS,
    LINE_IDS,
    SCOPES,
    estimate_temperature,
    get_calibration,
    subset_performance,
)


def verify(reference_root: Path) -> dict:
    # Suppress bytecode writes before importing any private module.
    sys.dont_write_bytecode = True
    from importlib.resources import files

    packaged = files("sc_teff_method").joinpath("data", CALIBRATION_ID)
    with tempfile.TemporaryDirectory(prefix="sc-teff-verify-") as tmp:
        projected = Path(tmp) / "export"
        hashes = export(reference_root, projected)
        for name in hashes:
            if projected.joinpath(name).read_bytes() != packaged.joinpath(name).read_bytes():
                raise AssertionError(f"Packaged projection differs from reference export: {name}")
    spec = importlib.util.spec_from_file_location(
        "_sc_teff_private_line_quality", reference_root / "src/scteff/line_quality.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    folder = reference_root / PREFIX
    coefficients = module.read_sealed(folder, "coefficients_all_methods.csv")
    subsets = module.read_sealed(folder, "subset_metrics.csv")
    max_temperature_difference = 0.0
    max_scatter_difference = 0.0
    estimates_checked = 0
    per_line_checked = 0

    def check(areas, configuration, metal):
        nonlocal max_temperature_difference, max_scatter_difference
        nonlocal estimates_checked, per_line_checked
        reference = module.estimate_temperature(
            areas, coefficients, configuration=configuration, metallicity=metal,
        )
        public = estimate_temperature(areas, configuration=configuration, feh=metal)
        if public.used_line_ids != reference["used_line_ids"]:
            raise AssertionError("Contributing line identities differ")
        differences = [abs(public.temperature_k - reference["temperature_k"])]
        for line in public.used_line_ids:
            differences.append(abs(
                public.line_temperatures_k[line] - reference["line_temperatures_k"][line]
            ))
            per_line_checked += 1
        max_temperature_difference = max(max_temperature_difference, *differences)
        if not all(math.isfinite(delta) and delta <= 1e-9 for delta in differences):
            raise AssertionError("Temperature disagrees with reference by more than 1e-9 K")
        if public.n_lines == 1:
            if public.line_scatter_k is not None or reference["line_spread_k"] is not None:
                raise AssertionError("Single-line scatter must remain undefined")
        else:
            delta = abs(public.line_scatter_k - reference["line_spread_k"])
            max_scatter_difference = max(max_scatter_difference, delta)
            if not math.isfinite(delta) or delta > 1e-9:
                raise AssertionError("Sample line scatter disagrees with reference")
        estimates_checked += 1

    illustrative = {
        "line1": 0.4, "line2": 0.35, "line3": 0.3, "line4": 0.65,
        "line5": 0.8, "line6": 0.5, "line7": 0.4, "line8": 0.4, "line9": 0.65,
    }
    coefficient_rows_checked = 0
    validation_rows_checked = 0
    for configuration in CONFIGURATIONS:
        calibration = get_calibration(configuration)
        for line in calibration.lines.values():
            row, = [r for r in coefficients if r["configuration"] == configuration
                    and r["measurement_source"] == "gaussian" and r["line_id"] == line.line_id]
            if (line.a_k_per_angstrom != float(row["a_k_per_angstrom"])
                    or line.c_k != float(row["c_k"])
                    or line.b_k_per_dex != (
                        float(row["b_k_per_dex"]) if row["b_k_per_dex"] else None)):
                raise AssertionError("Frozen coefficients changed")
            coefficient_rows_checked += 1
            lower, upper = line.training_strength_range_angstrom
            for area in (lower, (lower + upper) / 2, upper):
                check({line.line_id: area}, configuration, -0.2)
        for areas, metal in (
            ({"line1": 0.3, "line2": 0.27}, -0.2),
            ({"line4": 0.65, "line5": 0.8, "line6": 0.5}, -0.1),
            (illustrative, -0.1),
        ):
            check(areas, configuration, metal)
        for mask in range(1, 512):
            areas = {line: illustrative[line] for i, line in enumerate(LINE_IDS) if mask & (1 << i)}
            check(areas, configuration, -0.1)
        for scope in SCOPES:
            for cohort in COHORTS:
                selected = [r for r in subsets if r["model_id"] == calibration.model_id
                            and r["measurement_source"] == "gaussian"
                            and r["scope"] == scope and r["cohort"] == cohort]
                if len(selected) != 511:
                    raise AssertionError("Reference subset inventory is incomplete")
                for row in selected:
                    chosen = row["line_ids"].split("|")
                    reference = module.lookup_subset(
                        selected, chosen, model_id=calibration.model_id, scope=scope, cohort=cohort,
                    )
                    public = subset_performance(
                        chosen, configuration=configuration, scope=scope, cohort=cohort,
                    )
                    if (public.n_stars != int(reference["n_stars"])
                            or public.bias_k != float(reference["bias_k"])
                            or public.rmse_k != float(reference["rmse_k"])):
                        raise AssertionError("Held-out validation differs from reference lookup")
                    validation_rows_checked += 1
    return {
        "calibration_id": CALIBRATION_ID,
        "coefficient_rows_checked": coefficient_rows_checked,
        "estimates_checked": estimates_checked,
        "per_line_temperatures_checked": per_line_checked,
        "max_absolute_temperature_difference_k": max_temperature_difference,
        "max_absolute_scatter_difference_k": max_scatter_difference,
        "validation_rows_checked": validation_rows_checked,
        "validation_statistics_match_exactly": True,
        "packaged_export_matches_reference_byte_for_byte": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.reference_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
