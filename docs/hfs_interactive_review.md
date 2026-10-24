# Interactive Gaussian and HFS review (PR 6)

This optional local interface uses the validated Gaussian and HFS fitting
engines from PRs 2 and 5. It requires no external service or uploads.
All documentation examples use entirely analytical synthetic profiles.
No observational spectrum, spectral extract or template is shipped or used.

## Launch both independent scientific reviewers

Install the optional browser dependencies:

```sh
python -m pip install ".[inspect]"
```

Generate a local mathematical HFS spectrum, with invented profile parameters:

```sh
python examples/generate_hfs_spectrum.py --case noise --output-dir hfs-demo
sc-teff inspect hfs-demo/spectrum.csv --with-hfs \
  --delimiter , --header --wavelength-column wavelength --flux-column flux \
  --uncertainty-column uncertainty --mask-column exclude \
  --wavelength-unit angstrom --wavelength-frame air --continuum-normalised \
  --feh -0.2 --save-review gaussian-review.json \
  --save-hfs-review hfs-review.json
```

The interface binds only to the local loopback address `127.0.0.1` and
serves JavaScript locally. Choose **Gaussian** or **Hyperfine structure (HFS)**
from the method selector at the top. Both methods share the same immutable,
already prepared spectrum, but maintain separate fitting windows, masks,
manually chosen continuum adjustments and Pending/Accepted/Rejected decisions.

The HFS panels show the sum of all fixed hyperfine components, the individual
component positions, residuals and quality flags. They do **not** display a
single Gaussian as an HFS fit. Manual corrections are applied to an in-memory
working copy of the original flux and uncertainty, exactly once per refit.

Convergence is **never** an automatic scientific approval. Editing any accepted
line clears its approval for that measurement method, but does not alter the
other method's fit, controls or decision. Only explicitly accepted positive,
converged line measurements enter each method's reviewed temperature.

The summary bar shows the two **separate** reviewed temperature estimates.
They should not be averaged together or treated as independent calibrations:
both use the same stellar spectrum and reference scale, and their errors are
strongly correlated.

## Save, resume and reproduce

Click **Save review JSON** separately in each method's panel. Review files have
different formats:

- Gaussian review: `sc-teff-review` schema 2 from PR 4
- HFS review: `sc-teff-hfs-review` schema 1

The HFS review contains the frozen atomic/model identities, file checksum,
preparation/RV conventions, manually chosen windows and continuum anchors,
masks, decisions, annotations and scalar fit audits. It contains no spectrum
or wavelength-flux arrays. Both review formats can be loaded independently.

Resume both from the identical source spectrum:

```sh
sc-teff inspect hfs-demo/spectrum.csv --with-hfs \
  --review-file gaussian-review.json --hfs-review-file hfs-review.json
```

Or reproduce the accepted HFS result **without Bokeh**:

```sh
sc-teff replay-hfs hfs-demo/spectrum.csv --review-file hfs-review.json \
  --output-dir reproduced-hfs/
```

The HFS replay recalculates fits and the accepted-only temperature using the
frozen HFS coefficients. It rejects mismatched source-file and prepared-spectrum
checksums, invalid review identities and materially changed accepted fits.
It writes `review.json`, `summary.json`, `lines.csv` and HFS diagnostic plots.
Existing Gaussian `sc-teff replay` remains unchanged.

The optional `--with-hfs` flag leaves the earlier Gaussian-only inspection
workflow unchanged unless requested.

## Domain and measurement limitations

The reference HFS calibration's strict-domain sample selection uses a
**separately measured Gaussian-anchor area** of at least 0.08 Å. An HFS area
cannot substitute for this selection anchor. Interactive HFS reviews started
without independently supplied Gaussian anchors correctly report that strict
applicability is unknown. Displaying a frozen subset RMSE does not certify
applicability, nor does it validate manual continuum placement or post-hoc
line selection.

The HFS model fixes the 133 published-pattern component strengths and uses the
adopted A+B splitting. Only common amplitude, parent centre and component width
are fitted. The implementation uses **unweighted** least squares to reproduce
the approved reference method. Formal covariance omits continuum, blends,
instrument mismatch and atomic/model systematics.

This interface does not modify either frozen calibration, refit laboratory
constants, implement saturation/radiative transfer, or ingest private data.
