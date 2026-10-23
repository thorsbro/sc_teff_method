"""Run after installation: python examples/use_measured_areas.py.

These Gaussian areas are illustrative numbers, not observed measurements.
This example contains no spectra or wavelength–flux arrays.
"""

from sc_teff_method import estimate_temperature, subset_performance


def main() -> None:
    areas = {"line1": 0.30, "line2": 0.27}  # Gaussian absorption areas in Angstrom
    for configuration, feh in (("nandakumar_feh", -0.2), ("nandakumar_no_feh", None)):
        result = estimate_temperature(areas, configuration=configuration, feh=feh)
        performance = subset_performance(
            result.used_line_ids, configuration=result.configuration,
            scope="strict_strength_domain", cohort="exact_subset",
        )
        print(f"{configuration}: Teff = {result.temperature_k:.3f} K")
        print(f"  Lines ({result.n_lines}): {', '.join(result.used_line_ids)}")
        print(f"  Per-line temperatures (K): {dict(result.line_temperatures_k)}")
        if result.line_scatter_k is None:
            print("  Line-to-line sample scatter: undefined for one line")
        else:
            print(f"  Line-to-line sample scatter: {result.line_scatter_k:.3f} K (diagnostic)")
        print(
            f"  Held-out validation: N={performance.n_stars}, "
            f"bias={performance.bias_k:.3f} K, RMSE={performance.rmse_k:.3f} K"
        )
        print(f"  Calibration: {result.calibration_id}; model: {result.model_id}")
        for warning in result.warnings:
            print(f"  Warning [{warning.code}]: {warning.message}")
        print(f"  {result.uncertainty_note}")


if __name__ == "__main__":
    main()
