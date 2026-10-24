"""Export accepted-only HFS results without conflating Gaussian review products."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from .hfs_outputs import write_hfs_outputs
from .hfs_review import HFSReviewSession
from .review_io import protect_output_inputs


def write_hfs_review_outputs(
    session: HFSReviewSession,
    output_dir,
    *,
    overwrite: bool = False,
    plot_format: str = "png",
    spectrum_label: str = "Input spectrum",
    protected_paths=(),
):
    """Write HFS-reviewed products and human-readable, reproducible controls.

    Only explicitly accepted valid HFS fits contribute to the temperature.
    No spectral arrays are embedded in JSON or CSV. Input files are protected.
    """
    if not isinstance(session, HFSReviewSession):
        raise TypeError("HFS reviewed outputs require an HFSReviewSession")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    if plot_format not in ("png", "pdf", "both"):
        raise ValueError("Unsupported plot format")
    target = Path(output_dir).expanduser()
    protect_output_inputs(target, protected_paths)
    if target.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(target)
    if target.exists() and not overwrite:
        raise FileExistsError("Output directory exists; choose another or allow overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sc-teff-hfs-reviewed-", dir=target.parent) as td:
        stage = Path(td) / "products"
        files = write_hfs_outputs(
            session.reviewed, stage, plot_format=plot_format,
            spectrum_label=spectrum_label, input_metadata=session.to_dict()["input"],
        )
        session.save(stage / "review.json")
        summary_file = stage / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        summary["format"] = "sc-teff-hfs-reviewed"
        summary["review"] = {
            "format": "sc-teff-hfs-review",
            "schema_version": session.to_dict()["schema_version"],
            "notes": session.notes,
            "decisions": {k: {"state": v.state, "note": v.note}
                          for k, v in session.decisions.items()},
            "warning": "Numerical convergence is not scientific approval. Frozen validation "
                       "statistics do not validate post-hoc selection or manual continuum.",
        }
        for line, entry in summary["fits"].items():
            entry["review_state"] = session.decisions[line].state
            entry["review_note"] = session.decisions[line].note
            entry["uncorrected_area_angstrom"] = session.uncorrected_fits[line].area_angstrom
        summary_file.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
            encoding="utf-8",
        )
        table_path = stage / "lines.csv"
        with table_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            line = row["line_id"]
            row["review_state"] = session.decisions[line].state
            row["review_note"] = session.decisions[line].note
            row["uncorrected_area_angstrom"] = session.uncorrected_fits[line].area_angstrom
        with table_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        names = [*files, "review.json"]
        if target.exists():
            for name in names:
                os.replace(stage / name, target / name)
        else:
            stage.rename(target)
    return {name: target / name for name in names}
