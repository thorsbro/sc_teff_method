# Frozen numerical provenance

The public calibration identity is **`2026-10-05-nine-line-v1`**. Its authority is
the frozen 5 October 2026 nine-line snapshot in `thorsbro/scteff_paper`, a private,
read-only reference repository. No calibration has been refitted or updated.

The snapshot products were committed at
`1b748b220ec03aa9b6417ad3737f7542436672de`. The reference checkout audited for
this export was `ad70e2e8c9c6ac458168e862d609233e2c564af3`. The three source CSVs
and snapshot manifest are byte-identical at those revisions. Supporting guidance
was read at the audited checkout revision; the guidance file was added after
the original freeze. These revision identifiers establish provenance; no private
Git history is distributed.

The original full-sample fit identity is `2026-10-04-govind44-paper-v1`; the
leave-one-star-out validation identity is `2026-10-04-govind44-loocv-paper-v1`.
The nine-line snapshot preserves those fits and supplies the fixed-subset
validation. All nine lines are the main method. The historical four-line
Galactic-centre comparison does not define the package default.

## Source products and hashes

The source CSVs are under
`results/publication/2026-10-05-nine-line-v1/` in the private repository.

| Source file | SHA-256 |
| --- | --- |
| `coefficients_all_methods.csv` | `bfc0b011807f171e1bf8d14a47b559afd306d8ae27c205909f392e6c3cda330a` |
| `subset_metrics.csv` | `a340e1e79d92a0b0244a2162a3f683cef10726e25612efca52c5563e02cbfe30` |
| `line_quality.csv` | `21adaf4158078f815a1414b62e15d1a93051f0b033d25fa2373e360943b3ca83` |
| `metadata.json` | `51079148198763a987aae07bceb562ec1725eaceb4c53969da64970bbc66b508` |

The three CSV hashes were checked against the frozen manifest's
`output_sha256`. The public projection also records hashes of the reference
implementation, tests, `docs/nine_line_quality.md`, `config/lines.csv` and
`config/calibration_models.toml`. Their exact hashes, and the upstream generator
code hashes and software versions, are in the packaged
[provenance.json](../src/sc_teff_method/data/2026-10-05-nine-line-v1/provenance.json).
Source code and guidance are hashed for provenance; they are not copied into
the public package.

## Public projection

Only rows with `measurement_source="gaussian"` are exported. The model mapping is:

| Public configuration | Frozen model ID | Reference temperature scale |
| --- | --- | --- |
| `nandakumar_feh` | `legacy_direct_metallicity_0p08` | `nandakumar2023_igrins` |
| `nandakumar_no_feh` | `legacy_direct_1d_0p08` | `nandakumar2023_igrins` |

The versioned package data directory contains three files:

- `calibration.json`: 18 coefficient rows, nine per configuration, with line
  identity, air wavelength, training count, coefficients and marginal empirical
  coverage. The original coefficient strings are converted directly to binary64
  numbers, matching the reference implementation. No decimal precision is
  deliberately removed. Blank metallicity coefficients remain `null`, meaning
  the predictor is absent, rather than a coefficient of zero.
- `subset_metrics.csv`: 4,088 records, retaining the original sample-size, bias
  and RMSE strings verbatim. Each of two configurations has 511 subsets for
  both scopes and both cohorts. The compact `subset_mask` uses bit `i-1` for
  `linei`, so the non-empty masks are 1–511. A subset is an identity, not a rank.
- `provenance.json`: source identities and hashes, projection rules, upstream
  generation metadata and SHA-256 hashes of the two numerical products.

The packaged numerical hashes are:

| Public file | SHA-256 |
| --- | --- |
| `calibration.json` | `5aacb112f19ca739ba8c973622e77eff7f9e424095699bb8a59e7d6fee309c84` |
| `subset_metrics.csv` | `2c7618489d2849afb7d2e93899e607500427f8b62dc50b63bd4863d074b8231c` |
| `provenance.json` | `f9afdda035971d77a998ed00a82cd0ea7ada23cf720026905206b1b235c40cb6` |

The loader checks the provenance hash against a code anchor, then verifies
numerical products before their first use. Loaded immutable results are cached.
The public tests independently pin the file hashes and exercise edited-file
rejection. Scientific data updates require a new reviewed calibration identity.

No spectra, observed spectral plots, cutouts, wavelength–flux arrays, per-star
measurements or predictions, object lists, catalogues, personal paths or private
historical scripts are exported. The regression fixture contains illustrative
line-area inputs and the corresponding derived numerical predictions only.
Empirical range endpoints are approved aggregate calibration products.

## Meaning of the ranges and validation

The declared model-selection cut is inclusive Gaussian-anchor area
`W >= 0.08 Å`, with no upper strength or reference-temperature selection cut.
That declaration comes from `config/calibration_models.toml`; it is not inferred
from the empirical training minimum. The empirical strength ranges come from
training-included full-sample measurements in the coefficient table.

Temperature and metallicity coverage comes from the retained held-out
references in `line_quality.csv`. Both scopes have identical reference coverage
for these Gaussian models, which the exporter verifies. These are marginal
descriptive ranges, not formal validity intervals or a guarantee of joint
coverage. `line3` uses 43 stars and has narrower reference coverage; other lines
use 44. A metallicity-free prediction does not use metallicity, even though
reference-metallicity coverage can be checked diagnostically.

The subset statistics use saved leave-one-star-out predictions. Each star's
requested line temperatures are averaged, then compared with its reference.
Bias has the sign **prediction minus reference**. The exporter performs no
new fitting, weighting, clipping, subset selection or validation calculation.
It preserves the `strict_strength_domain`/`all_available` scopes and
`exact_subset`/`common_all9` cohorts. The public API returns N, bias and RMSE
from those frozen rows, without adding measurement uncertainties.

## Reproduce the export and comparison

These optional maintainer commands require access to the reference checkout.
They are not part of installation, public tests or GitHub Actions:

```bash
python tools/export_frozen_data.py \
    --reference-root PATH_TO_READ_ONLY_REFERENCE \
    --output-dir NEW_OUTPUT_DIRECTORY

python tools/verify_reference.py --reference-root PATH_TO_READ_ONLY_REFERENCE
```

The exporter reads only its explicit source-file allowlist, verifies the pinned
hashes and refuses an existing output directory. The result is deterministic.
The comparison tool first repeats that export into a temporary directory and
compares all three packaged files byte-for-byte. It then imports the private
fixed-coefficient estimator with bytecode writing disabled. The comparison
environment needs the reference implementation's NumPy dependency; the installed
public package does not.

Verification on Python 3.11 found:

- All 18 coefficients match their source binary64 values exactly.
- 1,082 estimates and 4,690 per-line temperatures agree within `1e-9 K`, covering
  individual-line training-range endpoints and midpoints, two- and three-line
  examples, the nine-line estimate and every one of the 511 subsets per model.
- The maximum temperature difference is `9.094947017729282e-13 K`; the maximum
  sample-scatter difference is `7.105427357601002e-15 K`. The public package uses
  standard-library averaging; these differences are floating-point effects.
- Every one of the 4,088 validation records has exactly matching N, bias and RMSE
  when compared with the private `lookup_subset` implementation.
- The re-extracted public products match the packaged files byte-for-byte.

The 60 fixed reference cases in `tests/fixtures/reference_regression.json` were
captured from that private estimator, using illustrative strengths and approved
aggregate training-range endpoints. No individual-star measurements are included. The
public suite also checks the temperature mean for all 511 subsets per model and
all 4,088 validation lookups. It has no reference-repository path or dependency.

## Citation and remaining verification

The software author's name in `CITATION.cff` is verified by the existing MIT
copyright notice and repository-owner metadata. The software is version 0.1.0
in this PR; no release date or DOI is claimed. The reference temperature scale
is [Nandakumar et al. (2023), A&A 675, A23](https://doi.org/10.1051/0004-6361/202346149),
whose article metadata was verified against the publisher. That article is
not presented as the new Sc I calibration paper.

TODO/VERIFY before a separate publication or data release: confirm the final
calibration-paper bibliographic metadata, author attribution, data-release
identifier and any additional numerical-data terms or co-author permissions.
The existing MIT software licence is unchanged; no additional data licence or
permissions have been inferred. See [DATA_NOTICE.md](../DATA_NOTICE.md).
