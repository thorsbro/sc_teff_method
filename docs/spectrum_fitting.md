# Gaussian fits to prepared spectra

This workflow measures user-supplied continuum-normalised spectra. No automatic
continuum fitting, vacuum conversion, unit inference or RV inference occurs.
Install `sc-teff-method[spectra]` from a checkout with `pip install ".[spectra]"`.
The original measured-area API still works with the dependency-free base install.

## Inputs and coordinates

`prepare_spectrum` accepts arrays; `load_spectrum` accepts whitespace ASCII and
explicitly delimited CSV. Both require `wavelength_unit="angstrom"`,
`wavelength_frame="air"` and `continuum_normalised=True`. These are declarations
by the user, not guarantees established by the loader. Other conventions are
rejected. FITS is deferred.

Wavelengths must be a one-dimensional, positive, finite, strictly increasing
array, with at least two pixels. Flux, optional absolute pixel standard deviations
and optional boolean masks must have equal lengths. Unmasked flux must be finite;
unmasked uncertainties must be positive and finite. Explicitly masked nonfinite
flux/uncertainty is allowed and excluded from fitting. The loader never silently
interpolates, sorts, fills gaps or rejects pixels. True means exclude in Python;
a file mask column uses `0=use, 1=exclude`. Unused extra file columns may contain
text. Selected columns must be numeric and rows must have consistent widths.

Examples:

```python
from sc_teff_method import load_spectrum, estimate_from_spectrum

spectrum = load_spectrum(
    "my_spectrum.csv", delimiter=",", header=True,
    wavelength_column="wave", flux_column="flux", uncertainty_column="error",
    mask_column="exclude", skiprows=0,
    wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
    rv_kms=25.0,
)
analysis = estimate_from_spectrum(spectrum, feh=-0.2)
```

None as RV means no correction, and zero is an explicit zero correction.
Positive RV denotes recession/redshift. Input wavelengths are divided by the
relativistic Doppler factor `sqrt((1+v/c)/(1-v/c))`, with `c=299792.458 km/s`.
The package does not resample flux, alter errors or perform barycentric/LSR/frame
corrections. Supply the correction appropriate to the actual wavelength grid.
Pass a prepared `Spectrum` directly to `estimate_from_spectrum` to avoid applying
the correction twice.

## Model, controls and provisional defaults

For each public Sc I wavelength, the model is

```text
F(lambda) = 1 - A exp[-(lambda-centre)^2/(2 sigma^2)]
W = sqrt(2*pi) A sigma
```

`A` is dimensionless absorption depth; `centre`, `sigma` and `W` use rest-frame
air Å. Sigma is the Gaussian standard deviation, not its FWHM. The area is
integrated over the entire analytic Gaussian, rather than a finite band.
Continuum is fixed to unity. Native pixels inside the inclusive window, after
all explicit masks, enter bounded least squares with an analytic Jacobian and
SciPy's trust-region reflective solver. The centre is optimised relative to the
window midpoint for numerical stability. See the
[SciPy solver documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

Default settings are generic mathematical choices, **not values measured from
observations or approved star-specific windows**:

| Control | Default |
| --- | --- |
| Window | Published rest wavelength ±3 Å |
| Allowed centre | Intersection of window with rest wavelength ±1 Å |
| Amplitude bounds | 0.000001–1 |
| Sigma bounds | 0.03–1.5 Å |
| Starting sigma | 0.35 Å; clipped inside bounds |
| Starting centre/depth | Deepest available eligible pixel / absorption depth; clipped inside bounds |
| Minimum used pixels | 7 |
| Maximum solver evaluations | 2000 |
| Solver tolerances | `ftol=xtol=gtol=1e-11` |
| Near-bound warning | Within 1% of allowed parameter interval |
| Residual RMS warning | Above 0.02 in normalised flux |
| Reduced chi-square warning | Above 3, when pixel errors supplied |
| Centre displacement warning | More than 0.5 Å from rest |
| Sampling warning | Fewer than four median pixel spacings per fitted FWHM |
| Raw missing/irregular pixel warning | Step larger than 1.5× median native spacing in the window |
| Internal raw gap warning | Step larger than 3× median native spacing |
| Edge coverage tolerance | 1.5× median native spacing |
| Covariance condition limit | Jacobian condition number 1e10 |
| Low-amplitude significance warning | Below 3× formal amplitude error or median supplied pixel error |

These checks are inspection prompts. Irregular but scientifically usable grids
can trigger warnings, and contaminated spectra can pass the numerical checks.
Intervals too far from the allowed centre range and invalid starting values are
input errors. Missing or insufficient usable pixels produce explicit unavailable
line outcomes, without stopping other lines.

Override windows, guesses and masks per line in Python:

```python
from sc_teff_method import FitOptions, GaussianParameters, estimate_from_spectrum

analysis = estimate_from_spectrum(
    spectrum, feh=-0.2,
    windows={"line4": (22021.5, 22026.7)},
    initial_parameters={"line4": GaussianParameters(0.4, 22024.2, 0.45)},
    excluded_regions={"line4": [(22025.0, 22025.4)]},
    excluded_lines=["line8"],
    options=FitOptions(rms_warning=0.015, sigma_bounds_angstrom=(0.04, 1.2)),
    quality_policy="exclude_warned",
)
```

These numbers are illustrative settings, not recommended windows for a star.
Windows and masks are absolute coordinates **after** RV correction. Per-line
window overrides receive `window_provisional=False` to distinguish caller choices
from package defaults; this does not scientifically approve those windows.

The equivalent command-line flags are `--window line4:22021.5:22026.7`,
`--initial line4:0.4:22024.2:0.45`, `--exclude-region line4:22025.0:22025.4`
and `--exclude-line line8`. Repeated mask intervals are supported. An optional
`--fit-config settings.json` accepts this structure:

```json
{
  "windows": {"line4": [22021.5, 22026.7]},
  "initial_parameters": {"line4": [0.4, 22024.2, 0.45]},
  "excluded_regions": {"line4": [[22025.0, 22025.4]]},
  "options": {"rms_warning": 0.015, "sigma_bounds_angstrom": [0.04, 1.2]}
}
```

Command-line window/initial overrides replace matching JSON entries; command-line
exclusion intervals are appended. Unknown keys, lines and invalid controls are
rejected. Default windows and every resolved per-line interval, initial override,
mask interval, parser setting and fit option are recorded in JSON output.

## Quality policy, temperatures and validation

`LineFit.status` is `converged`, `failed`, `unavailable` or `excluded`. Failure,
missing coverage and user exclusions have no fitted area or temperature. All nine
line identities remain in the fit table. Every converged result reports
`requires_inspection=True`, even if its numerical flags are empty.

Flags cover partial/one-sided coverage, raw gaps or missing/irregular pixels,
masked edges/gaps/line cores, insufficient samples, coarse FWHM sampling, bounds,
centre displacement, residual RMS, reduced chi-square, low amplitude significance,
solver failure and unavailable/ill-conditioned covariance. Inspect the `flags`
field and diagnostics, not just the status. Default-window provenance and the
general need for inspection are separate from numerical quality flags.

`quality_policy="exclude_warned"` is the default: only converged positive-area
fits with no numerical quality flags enter Teff. After inspection,
`quality_policy="include_warned"` (CLI `--include-warned`) explicitly permits
converged flagged fits. This never enables failed or unavailable lines, and user
line exclusions always take precedence. Calibration extrapolation warnings remain
separate: the original PR 1 API warns without discarding areas.

`analysis.line_temperatures_k` includes diagnostic temperatures for every
converged positive-area fit, even quality-flagged fits omitted from the combined
estimate. `analysis.temperature` contains only the included subset and is None
when no usable lines remain. `analysis.exclusion_reasons` and CSV's
`included_in_temperature` prevent confusing diagnostic values with contributions.

The workflow calls the unchanged `estimate_temperature` and `subset_performance`
interfaces. It averages line temperatures with equal weights. Both frozen
configurations are supported, including the independently fitted metallicity-free
model. Performance always uses `cohort="exact_subset"` for the actual included
line identity; `validation_scope` selects the existing strict or all-available
scope. The inclusive declared strength cut and empirical ranges remain distinct.
The JSON records whether the target's included areas pass that cut; strict-domain
statistics do not cover an included area below 0.08 Å. A lookup row is not a new
validation of the user's instrument, continuum choices, masks or quality policy.

## Formal errors and scientific limitations

Without pixel errors, fitting is unweighted and local covariance is estimated
from `(J.T J)^-1 * RSS/(N-3)`. With explicit absolute independent pixel errors,
residuals/Jacobian are divided by those errors and covariance is `(J.T J)^-1`
without residual rescaling. This weighting changes the noisy estimator from the
unweighted measurement procedure used for calibration and requires assessment
before scientific transfer. The covariance convention follows the standard
[absolute/relative sigma distinction](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).

Area variance uses the full covariance and gradient
`sqrt(2*pi) * (sigma, 0, A)`, including amplitude–sigma covariance. Formal errors
are unavailable at boundaries, with insufficient degrees of freedom, or when the
Jacobian is singular or poorly conditioned. Parameters may still be numerically
converged in those cases and carry flags. Formal errors are local linearisations
under the fixed continuum/Gaussian/noise assumptions. They exclude continuum
errors, blends, correlated pixels, profile mismatch and calibration uncertainty.

Continuum normalisation, fitting windows, blends and instrumental resolution
affect areas. Good numerical fits do not guarantee reliable temperatures. The
empirical calibration was established under the observational conditions described
in the manuscript. Held-out bias/RMSE remain relative empirical validation
statistics, not formal measurement uncertainties. Individual line errors are
correlated; line scatter, or scatter divided by sqrt(line count), is not the total
temperature uncertainty. Inspect fits before interpreting a combined temperature.

## Synthetic examples, figures and numerical verification

`examples/generate_synthetic_spectra.py` generates `clean`, `noise`, `missing`,
`blends`, `rv_offset` and `variable` CSV examples plus exact `truth.json`.
Noise uses independent NumPy draws with default seed **20261023**, sigma 0.003.
Amplitudes, widths, centres, blends, continuum offsets and sampling are entirely
invented mathematical choices defined in `synthetic.py`; only the published line
centres come from the public calibration. These toy profiles are not stellar
spectral synthesis and do not reproduce cool-giant atmospheric complexity.

Reproduce the committed synthetic figure in a fresh directory:

```bash
python examples/fit_synthetic_spectrum.py --case blends \
  --mask-illustrative-blend --output-dir synthetic_diagnostic_example
```

The figure has nine profile/residual panels, including fitted and rest centres,
fitting-window boundaries, excluded pixels, area and status. Masked nonfinite
pixels are marked at continuum unity in the profile panel because no flux value
exists; no residual is invented for them. Local user outputs can use the user's
own spectrum; the repository figure is entirely analytical.

All fitting tests create analytical Gaussian arrays. Independent calculations
check centre/depth/sigma/area recovery, profile quadrature, Jacobian finite
differences, covariance/error propagation, both RV signs, mask/gap/coverage
behaviour and PR 1 temperatures. One independent nine-line noiseless sweep found
maximum area error `2.35e-13 Å` and combined-temperature difference `3.32e-11 K`.
Tolerances in tests allow appropriate floating-point/noise effects. Noise tests
use known errors and fixed seeds. These establish numerical correctness, not
observational accuracy or transfer accuracy for arbitrary instruments.

The only private reference consulted for this PR was numerical code in
`src/scteff/fitting.py`, SHA-256
`a69a39188c1160d0f5f86f875cc9ac0f1d298d11519b2e8e7d72661584eaba08`.
It was used to check model, solver, area and covariance definitions. No private
spectra, extracts, observed arrays, star-specific windows, continuum corrections,
per-star results or observational derivatives were read or used. There was no
parity test using observational inputs. The original frozen public calibration
products and their provenance, licence and citation metadata remain unchanged.

TODO/VERIFY: scientifically assess windows, diagnostic thresholds, continuum
preparation and instrument/profile transfer for the intended user's data. No new
observational validation is claimed. The final paper citation/data-release terms
already listed in the numerical provenance remain pending. FITS, automatic
continuum normalisation, HFS, rectangular bands, interactive continuum editing,
stellar spectral synthesis and calibration refitting remain deferred.
