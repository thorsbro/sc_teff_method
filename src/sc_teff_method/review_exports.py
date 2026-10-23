"""Reviewed products built from the existing PR 2 output and plotting functions."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .api import estimate_temperature
from .diagnostics import plot_diagnostics
from .outputs import write_fit_outputs
from .review_io import protect_output_inputs


def write_review_outputs(
    session,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    plot_format: str = "png",
    spectrum_label: str = "Input spectrum",
    protected_paths=(),
) -> dict[str, Path]:
    """Export review.json, summary.json, lines.csv and reviewed diagnostic figures.

    The temperature is constructed by ReviewSession from explicitly accepted
    fits using PR 1; no numerical fitting or calibration is performed here.
    Existing directories require explicit overwrite, and inputs are protected.
    """
    from .review import ReviewSession

    if not isinstance(session, ReviewSession):
        raise TypeError("session must be a ReviewSession")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if plot_format not in ("png", "pdf", "both"):
        raise ValueError("plot_format must be png, pdf or both")
    destination = Path(output_dir).expanduser()
    protect_output_inputs(destination, protected_paths)
    if destination.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {destination}; use overwrite=True")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = session.reviewed
    review = session.to_dict()
    formats = ("png", "pdf") if plot_format == "both" else (plot_format,)
    with tempfile.TemporaryDirectory(
        prefix=".sc-teff-review-", dir=destination.parent
    ) as temporary:
        products = Path(temporary) / "products"
        # This is the unmodified PR 2 numerical export; annotate scientific review separately.
        files = write_fit_outputs(
            result,
            products,
            plot_format=plot_format,
            spectrum_label=spectrum_label,
            input_metadata=review["input"],
        )
        session.save(products / "review.json")
        summary_path = products / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["estimate_kind"] = "explicitly_reviewed"
        summary["schema_version"] = 2
        summary["continuum_options"] = asdict(session.continuum_options)
        summary["continuum_note"] = (
            "Entirely manual local flux/uncertainty division by C, starting from the original "
            "input for every refit. The Gaussian continuum stays fixed at unity. Formal "
            "covariance excludes continuum-placement uncertainty. Frozen calibration RMSE "
            "does not validate these corrections or post-hoc selection."
        )
        summary["review"] = {
            "format": review["format"],
            "schema_version": review["schema_version"],
            "notes": review["notes"],
            "decisions": {line: values["decision"] for line, values in review["lines"].items()},
            "selection_note": (
                "Frozen subset statistics do not validate arbitrary post-hoc manual line "
                "selection or newly chosen continuum corrections."
            ),
        }
        uncorrected_areas = {
            line: fit.area_angstrom
            for line, fit in session.uncorrected_fits.items()
            if fit.status == "converged" and fit.area_angstrom is not None and fit.area_angstrom > 0
        }
        uncorrected_temperatures = (
            estimate_temperature(
                uncorrected_areas,
                configuration=session.configuration,
                feh=session.feh,
                reference_temperature_k=session.reference_temperature_k,
            ).line_temperatures_k
            if uncorrected_areas else {}
        )
        for line, fit in summary["fits"].items():
            fit["review_state"] = session.decisions[line].state
            fit["review_note"] = session.decisions[line].note
            fit["continuum"] = asdict(session.controls[line].continuum)
            fit["uncorrected_area_angstrom"] = session.uncorrected_fits[line].area_angstrom
            fit["uncorrected_line_temperature_k"] = uncorrected_temperatures.get(line)
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        table_path = products / "lines.csv"
        with table_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            line = row["line_id"]
            decision = session.decisions[line]
            row["review_state"] = decision.state
            row["review_note"] = decision.note
            row["accepted_with_quality_flags"] = decision.state == "accepted" and bool(
                session.fits[line].flags
            )
            correction = session.controls[line].continuum
            row["continuum_enabled"] = correction.enabled
            row["continuum_mode"] = correction.mode
            row["continuum_constant_level"] = correction.level
            row["continuum_anchors"] = json.dumps(correction.anchors, allow_nan=False)
            row["uncorrected_area_angstrom"] = session.uncorrected_fits[line].area_angstrom
            row["uncorrected_line_temperature_k"] = uncorrected_temperatures.get(line)
        with table_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        figure = plot_diagnostics(
            result,
            spectrum_label=spectrum_label,
            line_spectra={line: session.corrected_spectrum(line) for line in session.fits},
            continuum_corrections={
                line: control.continuum for line, control in session.controls.items()
            },
        )
        try:
            for index, (line, fit) in enumerate(session.fits.items()):
                state = session.decisions[line].state
                area = "missing" if fit.area_angstrom is None else f"{fit.area_angstrom:.5f} Å"
                temperature = result.line_temperatures_k.get(line)
                value = "missing" if temperature is None else f"{temperature:.1f} K"
                correction = session.controls[line].continuum
                continuum_label = correction.mode if correction.active else "unity"
                figure.axes[2 * index].set_title(
                    f"{line}: {fit.status} / {state} / C={continuum_label}"
                    f"\nW={area}; line Teff={value}",
                    fontsize=8.5,
                )
            figure.texts[-1].set_text(
                "Manual review states are separate from numerical flags. "
                "Frozen RMSE does not validate post-hoc selection or continuum adjustments.\n"
                "Blue pixels/residuals use corrected flux. Formal errors omit continuum placement."
            )
            for extension in formats:
                figure.savefig(products / f"diagnostics.{extension}", dpi=150)
        finally:
            figure.clear()
        names = [*files, "review.json"]
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Output directory appeared during writing: {destination}")
            for name in names:
                os.replace(products / name, destination / name)
        else:
            products.rename(destination)
    return {name: destination / name for name in names}
