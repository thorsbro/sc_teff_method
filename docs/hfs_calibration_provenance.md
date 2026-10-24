# Frozen HFS temperature calibration

The HFS calibration identity remains **`2026-10-05-nine-line-v1`**, the frozen
5 October 2026 nine-line publication snapshot. No stellar calibration is refitted.
The HFS coefficients and validation rows are separate numerical products from
the existing Gaussian package data. The adopted production atomic pattern is
`hala2022_ab_generated`.

## Method identity and numerical products

Use `get_hfs_calibration`, `estimate_hfs_temperature` and
`hfs_subset_performance` for HFS areas. The ordinary `get_calibration`,
`estimate_temperature` and `subset_performance` functions retain their Gaussian
meaning. The numerical model IDs in the private frozen tables are shared across
measurement methods; preserving `measurement_source="hfs"` is therefore essential.
Do not infer the profile method from the model ID alone.

| Public configuration | Frozen model ID | Temperature scale |
| --- | --- | --- |
| `nandakumar_feh` | `legacy_direct_metallicity_0p08` | `nandakumar2023_igrins` |
| `nandakumar_no_feh` | `legacy_direct_1d_0p08` | `nandakumar2023_igrins` |

The HFS package data directory is
`src/sc_teff_method/data/2026-10-05-nine-line-v1-hfs/`. It contains:

- `calibration.json`: the 18 supported HFS coefficient rows, marginal empirical
  HFS training-strength ranges, retained held-out reference-temperature and
  metallicity ranges, and original full-sample/validation identities.
- `subset_metrics.csv`: 4,088 HFS validation rows, comprising all 511 non-empty
  subsets for both configurations, both scopes and both cohorts. Bit `i-1` of
  `subset_mask` denotes `linei`; masks run from 1 through 511.
- `provenance.json`: the permitted frozen-source hashes, original generation
  metadata, projection rules and hashes of the two derived numerical files.

Coefficient decimal strings are converted directly to binary64, matching the
reference arithmetic. No decimal precision is intentionally removed. Missing
metallicity coefficients remain `null`; they are not replaced with zero.
Sample sizes, biases and RMSE strings are copied verbatim. The original fit
snapshot is `2026-10-04-govind44-paper-v1`; the leave-one-star-out snapshot is
`2026-10-04-govind44-loocv-paper-v1`.

| Public HFS product | SHA-256 |
| --- | --- |
| `calibration.json` | `b62db1c8b3700cbd082664eda4e98c65a6f76fd3ea6113ef34d14395eb184ed3` |
| `subset_metrics.csv` | `a9e43b571d78365be83f96702212c1b688ff5207398f1d98fff8d4f887a9c906` |
| `provenance.json` | `f838b0eadba36ff5d26df2edc291a719424bc1acd635ca5b970929e0a1e68833` |

The loader pins the provenance digest and verifies each numerical product
before caching immutable results. Scientific data changes require a separately
reviewed numerical version; the Gaussian data remain unchanged.

## The strict-domain Gaussian anchor

The frozen shared comparison selected stars/lines using an inclusive
**Gaussian-anchor area `W_Gaussian >= 0.08 Å`**, with no upper cut. That is not
a selection on the fitted HFS area. The HFS training-strength ranges are a
different aggregate description and must not be substituted for the selection.

```python
from sc_teff_method.hfs_calibration import estimate_hfs_temperature, hfs_subset_performance

# These are invented illustrative measurements, not observed stellar values.
result = estimate_hfs_temperature(
    {"line1": 0.30, "line2": 0.27}, feh=-0.2,
    gaussian_anchor_areas={"line1": 0.10, "line2": 0.09},
)
print(result.temperature_k)  # 3642.873079703434
print(result.strict_strength_domain_applicable)  # True
performance = hfs_subset_performance(result.used_line_ids)
print(performance.n_stars, performance.bias_k, performance.rmse_k)
```

Separate Gaussian anchors affect applicability metadata only; they never enter
the HFS temperature prediction. The result's `strict_strength_domain_applicable`
is `True` only when every selected line has a passing Gaussian anchor, `False`
when at least one known anchor fails, and `None` when required anchors are absent
and no known failure already establishes exclusion. Missing anchors produce
explicit returned warnings. An HFS area below 0.08 Å is not by itself evidence
of failure of the Gaussian-anchor cut. Empirical HFS range warnings still apply.

`hfs_subset_performance` returns a frozen lookup, not a claim of target
applicability. `strict_strength_domain` uses the historical Gaussian-anchor
selection; `all_available` removes that selection cut. `exact_subset` uses the
available held-out cohort for the requested fixed identity; `common_all9` uses
the complete-nine-line cohort for every subset of a model/scope. The lookup
reports the original sample size, bias (prediction minus reference) and RMSE.

These are empirical predictive statistics on the reference scale. They are
not formal per-star errors or a validated uncertainty for arbitrary instruments,
new continuum choices, blends or post-hoc line-selection procedures. Individual
line errors are correlated; the line-to-line sample scatter is not the total
temperature uncertainty. No additional total error bar is invented.

## Read-only extraction and source hashes

Only the following four files are read from
`thorsbro/scteff_paper/results/publication/2026-10-05-nine-line-v1/`:

| Approved source | SHA-256 |
| --- | --- |
| `coefficients_all_methods.csv` | `bfc0b011807f171e1bf8d14a47b559afd306d8ae27c205909f392e6c3cda330a` |
| `subset_metrics.csv` | `a340e1e79d92a0b0244a2162a3f683cef10726e25612efca52c5563e02cbfe30` |
| `line_quality.csv` | `21adaf4158078f815a1414b62e15d1a93051f0b033d25fa2373e360943b3ca83` |
| `metadata.json` | `51079148198763a987aae07bceb562ec1725eaceb4c53969da64970bbc66b508` |

The sealed source snapshot was committed at
`1b748b220ec03aa9b6417ad3737f7542436672de`. The CSV digests are checked against
`metadata.json` as well as code-pinned values. The Gaussian-anchor cut definition
is already documented in the public frozen Gaussian provenance; no additional
private calibration configuration or observational file is read for this export.

The optional maintainer command reads exactly that allowlist, verifies every
digest and writes a new directory outside the read-only reference repository:

```bash
python tools/export_hfs_calibration.py \
  --reference-root PATH_TO_READ_ONLY_REFERENCE \
  --output-dir NEW_HFS_NUMERICAL_DIRECTORY
```

There is no reference-package import, directory traversal, Git-history copy,
calibration fit or computation of new stellar validation statistics. No spectra,
individual-star measurements, predictions, object identities, catalogues or
observed spectral plots are exported. Public tests run from the packaged
products and fixed illustrative areas without access to the private repository.

The software remains MIT-licensed. With co-author agreement, the released
Gaussian and HFS empirical calibration products are licensed under CC BY 4.0,
as described in [DATA_NOTICE.md](../DATA_NOTICE.md). Selected third-party
atomic data retain their separately attributed CC BY 4.0 terms. The
`v1.0.0` software release is archived on
[Zenodo (DOI: 10.5281/zenodo.23621161)](https://doi.org/10.5281/zenodo.23621161).
The accompanying scientific paper's journal citation remains pending.
