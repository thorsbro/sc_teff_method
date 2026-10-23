# sc_teff_method

Calculate effective temperatures of cool giant stars from measured K-band Sc I
Gaussian line areas. This small Python package applies the frozen
**5 October 2026 nine-line calibration** on the Nandakumar et al. (2023)
temperature scale. It has no runtime dependencies beyond Python 3.11 or later.

Supply any non-empty subset of `line1` through `line9`. Each line gives a
temperature; the combined estimate is their **unweighted mean**. The main method
uses up to all nine lines. The historical four-line Galactic-centre comparison
is not a default configuration.

## Install

From a checkout of this repository:

```bash
python -m pip install .
python examples/use_measured_areas.py
```

The installed package includes its calibration and validation tables. It runs
offline and needs no access to the private paper repository. This PR does not
publish a PyPI package, GitHub release or archive deposit.

## Use your measured areas

```python
from sc_teff_method import estimate_temperature, subset_performance

# Illustrative Gaussian areas in Angstrom; these are not observed measurements.
result = estimate_temperature(
    {"line1": 0.30, "line2": 0.27},
    configuration="nandakumar_feh",
    feh=-0.2,
)

print(result.temperature_k)          # 3648.659937687841
print(dict(result.line_temperatures_k))
print(result.used_line_ids)          # ('line1', 'line2')
print(result.n_lines)                # 2
print(result.line_scatter_k)         # 30.880647035312226; diagnostic only
print(result.calibration_id)         # '2026-10-05-nine-line-v1'
for warning in result.warnings:
    print(warning.code, warning.line_id, warning.message)

performance = subset_performance(
    result.used_line_ids,
    configuration=result.configuration,
    scope="strict_strength_domain",
    cohort="exact_subset",
)
print(performance.n_stars)           # 44
print(performance.bias_k)            # -1.0468992999376239
print(performance.rmse_k)            # 51.379990357673506
```

These validation statistics are **not an error bar** on the example temperature.
Bias means Sc I prediction minus reference temperature. RMSE is the root mean
square of those residuals, including the bias.

Choose the calibration that matches the information available:

| Configuration | Frozen equation for each line | [Fe/H] input |
| --- | --- | --- |
| `nandakumar_feh` | `T = a * W + b * [Fe/H] + c` | Required, in dex |
| `nandakumar_no_feh` | `T = a * W + c` | Optional for coverage checks only |

The metallicity-free relation was fitted independently. Setting [Fe/H] to zero
in `nandakumar_feh` does not give that relation.

```python
without_feh = estimate_temperature(
    {"line1": 0.30, "line2": 0.27},
    configuration="nandakumar_no_feh",
)
print(without_feh.temperature_k)     # 3623.3789081217
```

`W` must be the positive analytic absorption area of a Gaussian fitted to a
continuum-normalised line, in **Å**, with the area definition used by the
calibration. For a Gaussian absorption depth `A` and width `sigma` in Å,
`W = sqrt(2*pi) * A * sigma`, integrated over the full Gaussian. Finite-band
equivalent widths, HFS-fit areas and other measurement definitions are not
interchangeable with these coefficients. Convert mÅ to Å before calling the API.

Omit unavailable lines from the mapping. Zero, `None`, blank strings and NaN
are rejected; none is used as a missing-value substitute. Booleans, complex
numbers, unknown IDs, empty input and non-finite values are also rejected.
The API uses every supplied positive, finite area, including extrapolations;
it never clips values or silently discards lines. Results list contributing
lines in canonical order.

| Line ID | Air wavelength (Å) | Line ID | Air wavelength (Å) | Line ID | Air wavelength (Å) |
| --- | ---: | --- | ---: | --- | ---: |
| `line1` | 21730.467 | `line4` | 22024.168 | `line7` | 22266.719 |
| `line2` | 21812.169 | `line5` | 22051.984 | `line8` | 22394.703 |
| `line3` | 21842.781 | `line6` | 22065.302 | `line9` | 22656.776 |

## Applicability and warnings

The **declared Gaussian-strength selection cut is inclusive `W >= 0.08 Å`**,
with no declared upper strength cut. It is separate from each line's empirical
full-sample training-strength range. Passing that cut does not establish that
an area is within the training range. For example, `line1=0.08` passes the
selection cut but lies below its empirical training minimum of about 0.1016 Å.

`estimate_temperature` returns separate warnings for areas below the declared
cut and areas outside the empirical training range. It also checks the empirical
reference-temperature and metallicity coverage for each supplied line. These
are marginal coverage checks, not a guarantee of joint support or validity.

The retained reference temperatures span 3347–3800 K and [Fe/H] −0.92 to +0.25
for most lines. `line3` has narrower coverage: 3347–3742 K and −0.66 to +0.25 dex.
When an independent temperature is available, pass `reference_temperature_k`
to check that value. Otherwise each estimated line temperature is checked as
an **applicability proxy**, not a known reference temperature. This check never
controls line availability. The metallicity-free configuration can check
metallicity coverage only if the optional `feh` is supplied; that value never
enters its temperature equation.

Inspect the exact per-line coefficients and ranges with:

```python
from sc_teff_method import get_calibration

calibration = get_calibration("nandakumar_feh")
print(calibration.lines["line3"])
```

Warnings have stable `code`, `line_id` and `message` attributes. Codes are
`outside_declared_strength_cut`, `outside_training_strength_range`,
`outside_reference_temperature_range` and `outside_reference_feh_range`.
They are returned in `result.warnings`, not emitted by Python's warnings module.

Applicability also depends on the reference scale, continuum definition,
resolving power, blends and measurement procedure. The source calibration uses
IGRINS measurements of cool giants. This API does not validate or correct a
new instrument, measurement procedure or stellar population.

## Subset validation and uncertainty

The package contains all **511 non-empty fixed subsets**, for both
configurations, two scopes and two cohorts: 4,088 validation records.

| Option | Meaning |
| --- | --- |
| `scope="strict_strength_domain"` | Retain held-out predictions passing the declared Gaussian-anchor strength cut. |
| `scope="all_available"` | Retain finite held-out predictions irrespective of that cut. |
| `cohort="exact_subset"` | Include every reference star with all requested lines usable. |
| `cohort="common_all9"` | Evaluate every subset on the same complete-nine-line reference cohort for that model and scope. |

Both scopes happen to give identical statistics in this frozen Gaussian sample;
their definitions remain distinct. Neither scope imposes the empirical training
ranges. `scope` and `cohort` select validation records; they do not filter a
target's supplied areas. If a target area is below 0.08 Å, strict-domain
validation does not cover that measurement. `all_available` is not evidence
of accuracy for arbitrary extrapolations.

Look up the actual contributing line identity for your target. Exact subsets
containing `line3` have 43 held-out stars; other exact subsets have 44.
Every `common_all9` cohort has 43 stars. Exact-nine validation therefore differs
from a population estimate that uses eight or nine lines per star. For example,
the exact-nine metallicity-dependent RMSE is about 48.95 K on 43 stars; the
reference variable-availability result is about 50.19 K on 44 stars. The latter
is documented for context and is not the lookup result for `line1`–`line9`.

Validation uses saved leave-one-star-out predictions. Bias and RMSE are
**empirical predictive statistics relative to the Nandakumar reference scale**,
not formal measurement uncertainties, absolute temperature accuracy or universal
per-star uncertainties. The line errors are correlated. The sample standard
deviation of the line temperatures (`line_scatter_k`, `ddof=1`) measures only
their internal disagreement; it is `None` for one line. Neither that scatter
nor scatter divided by `sqrt(n_lines)` gives the total uncertainty in the mean.
The API does not propagate measurement errors or fit a calibration confidence
interval. Do not select a supposedly best subset after inspecting these RMSE
values without separate selection validation.

## Provenance, citation and scope

The frozen coefficients are unchanged. The compact numerical export records
the source snapshot, source revisions, SHA-256 hashes and projection rules in
[docs/provenance.md](docs/provenance.md) and the packaged `provenance.json`.
Only the two Gaussian configurations and their validation summaries are shipped.

[CITATION.cff](CITATION.cff) identifies this software using the verified repository
author. The reference temperature scale is described by
[Nandakumar et al. (2023), A&A 675, A23](https://doi.org/10.1051/0004-6361/202346149).
That reference describes the temperature scale, not a new DOI for this Sc I
calibration. The final calibration-paper citation and public data-release
identifier remain TODO/VERIFY; neither is invented here.

The [MIT software licence](LICENSE) is retained. See
[DATA_NOTICE.md](DATA_NOTICE.md) for the limited numerical export and pending
data-attribution/permission checks. No additional data licence or co-author
permissions are assumed.

PR 1 accepts measured areas only. Spectrum reading, continuum normalisation,
radial-velocity correction, Gaussian fitting, HFS fitting, synthetic spectral
generation and calibration refitting are outside its scope. There are **no
spectral datasets**, observed spectral plots, cutouts or wavelength–flux arrays
in this package. Examples and tests use illustrative areas and approved derived
calibration products only.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m build
```

Public tests run without the private repository. They include fixed regressions
from the reference estimator, every subset identity and option combination,
all subset temperature means, input validation, boundary warnings and data
integrity checks. GitHub Actions builds distributions and runs the tests and
example against the installed wheel on Python 3.11 and 3.14.
