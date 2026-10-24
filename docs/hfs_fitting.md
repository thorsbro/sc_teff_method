# HFS fitting and temperature estimation

Install from the public checkout with `python -m pip install ".[spectra]"`.
The numerical HFS engine uses the existing NumPy, SciPy and Matplotlib optional
dependencies; it requires no Bokeh, network service or private repository.
The original Gaussian commands, review files and coefficients retain their
existing meaning. Interactive HFS inspection is deferred to a subsequent PR.

## Start with an analytical nine-line spectrum

```bash
python examples/generate_hfs_spectrum.py --output-dir hfs-demo --case noise
sc-teff fit-hfs hfs-demo/spectrum.csv \
  --delimiter , --header --wavelength-column wavelength --flux-column flux \
  --uncertainty-column uncertainty --mask-column exclude \
  --wavelength-unit angstrom --wavelength-frame air --continuum-normalised \
  --feh -0.2 --output-dir hfs-results --plot-format both \
  --label "Analytical synthetic HFS spectrum"
```

The generator writes `spectrum.csv` and `truth.json` using independent analytical
component Gaussians, invented amplitudes/widths/parent shifts and reproducible
noise. The default seed is **20261025**. Only the public frozen atomic pattern is
reused; no observed spectrum, stellar fit or continuum template supplies the
example parameters. The `noise` case has independent Gaussian flux noise with
standard deviation 0.0025 and a corresponding uncertainty column.

Other generator cases are `clean`, `missing`, `blends`, `rv_offset` and
`variable`. They exercise absent lines, invented neighbours/masks, an explicit
raw-grid redshift and different sampling/widths. The RV example requires
`--rv-kms 67`; the package does not infer this correction.

![Entirely analytical HFS diagnostic figure](images/synthetic_hfs_diagnostics.png)

These simplified profiles establish numerical correctness. They do not model
the full complexity of a cool-giant atmosphere or establish instrumental
transfer accuracy.

## Prepare your own input

Use the same ASCII/CSV conventions as the
[Gaussian fitting workflow](spectrum_fitting.md). Explicitly supply increasing
**air wavelengths in Å** and approximately continuum-normalised flux. Named
columns require `--header`; numerical column indices are zero based. Delimiters,
pixel masks and optional uncertainties are configurable. Mask value `1` means
exclude, `0` means use. Nonfinite flux/errors require an explicit input mask.
Coverage loss, missing pixels and sparse sampling remain visible in diagnostics.

No unit, air/vacuum convention, continuum or velocity correction is inferred.
Positive radial velocity means recession/redshift:

```text
lambda_rest = lambda_input / sqrt((1+v/c)/(1-v/c)), c = 299792.458 km/s.
```

All parent centres, fitting windows, masks and continuum anchors subsequently
use the corrected **rest-frame air-Å** coordinates. The existing preparation
function applies this correction once, without resampling flux. A prepared
`Spectrum` should be passed directly to the HFS functions without another RV
correction.

## Fixed atomic pattern and three fitted parameters

The adopted dataset is **`hala2022_ab_generated`**, with Sc-45 nuclear spin
I = 7/2. The component counts for lines 1–9 are
**21, 16, 9, 12, 21, 18, 6, 12, 18**, totalling 133. Each line uses

```text
F(lambda) = 1 - A sum_k s_k exp[-(lambda - lambda0 - delta_lambda_k)^2 / (2 sigma^2)]
sum_k s_k = 1
W_HFS = sqrt(2*pi) A sigma.
```

Only common amplitude A, parent centre λ₀ and common component sigma are free.
A is the coefficient of the normalised mixture, not the central depth of a
single Gaussian. Sigma is the shared component standard deviation in Å, not the
overall HFS-pattern width or a FWHM. The area integrates the complete analytical
profile, including tails outside the fitted window. Judge the available coverage
before interpreting that extrapolation.

Laboratory A/B constants, component positions, F assignments and relative
strengths are fixed. No individual component, continuum coefficient, isotope
mixture or saturation parameter is optimised. Electric-dipole selection rules
and independent Wigner 6-j calculations verify the component assignments.

Production positions include both magnetic-dipole A and electric-quadrupole B.
For this nine-transition selection, the published Hala & Nave Table 4 positions
behave approximately as A-only calculations; those positions remain comparison
metadata. They do not replace the A+B production pattern. Line9 retains the
positive-A upper-level assignment. B = 0 where J < 1 is physically distinct from
the explicit B = 0 assumption for three levels with unreported B.

Production strengths retain the frozen twelve-decimal normalised versions of
the published six-decimal strengths, followed by the reference loader's final
per-line normalisation. They are not replaced by exact Wigner strengths.
Relative wavelength offsets use the stable exact relation
`delta_lambda = -lambda_parent * delta_wavenumber / (atomic_cog + delta_wavenumber)`.
The atomic COG wavenumber is independent of the canonical air parent; it is not
calculated from `1e8/lambda_air`. See the
[atomic-data guide](hfs_atomic_data.md) for constants, source hashes, missing-B
statuses, Table 4 comparisons and CC BY 4.0 attribution.

## Windows, masks, convergence and formal errors

The existing bounded least-squares HFS reference engine uses an analytical
Jacobian and relative-centre conditioning. The reproduction baseline is
**unweighted**. Supplying pixel uncertainties enables a separately labelled
reduced-chi-square diagnostic and consistent continuum scaling, but does not
change the fit weights. Weighted HFS fitting is not implemented; a non-unweighted
Python `weighting` argument is rejected. This differs from the Gaussian workflow,
where supplied uncertainties enable weighting.

Provisional public defaults are:

| Control | Default |
| --- | --- |
| Window | Canonical air parent ±3 Å |
| Parent-centre bounds | Parent ±1 Å, intersected with the window |
| Common amplitude bounds | 0–1 |
| Component sigma bounds | 0.03–1.5 Å |
| Minimum unmasked samples | 10 |
| Maximum function evaluations | 5,000 |
| Residual-RMS warning | Above 0.03 |
| Boundary-warning fraction | 0.001 of the parameter range |
| Jacobian-condition warning | Above 10⁶ |

These mathematical defaults are not windows copied from individual stars or
observationally approved instrument settings. Use `--window LINE:LOWER:UPPER`,
`--initial LINE:AMPLITUDE:PARENT_CENTRE:SIGMA`,
`--exclude-region LINE:LOWER:UPPER`, `--exclude-line LINE` and the existing JSON
`--fit-config` structure for explicit overrides. Python callers can supply
`HFSFitOptions`, `HFSParameters` and per-line mappings.

Every fit records convergence, residual RMS, sample/exclusion counts, parent
displacement, width and bounds. Coverage, gaps, sparse component-Gaussian FWHM
sampling, ill-conditioned covariance and high residuals have explicit flags.
`hfs_components_cross_window` or `hfs_components_near_window_edge` warns about
component centres crossing or approaching a window boundary. Gaussian tails
outside the window are expected; these flags specifically concern component
centres. A small formal area error does not excuse a truncated pattern.

Formal covariance is residual-scaled unweighted least-squares covariance. Area
uncertainty includes amplitude-width covariance:

```text
Var(W_HFS) = 2*pi [sigma^2 Var(A) + A^2 Var(sigma) + 2 A sigma Cov(A,sigma)].
```

It holds the atomic pattern and continuum fixed and excludes correlated-noise,
continuum-placement, blend, profile and calibration systematics. Boundary or
singular/ill-conditioned fits can have unreliable or missing formal errors.
Neither this error, line scatter nor held-out RMSE is an unvalidated total
temperature uncertainty.

Numerical convergence is not scientific acceptance. Failed/unavailable lines
never contribute. The default policy excludes converged measurements carrying
quality flags; after inspection `--include-warned` explicitly permits their
inclusion with flags retained. `--exclude-line` always excludes the selected
measurement. If no usable lines remain, temperature and subset performance are
missing; diagnostics are still written and the command exits with status 3.
Invalid input/configuration exits with status 2.

## Python fitting and measured-area APIs

For a prepared spectrum, use `fit_hfs_lines(spectrum)` to obtain all nine HFS
measurement outcomes or `estimate_hfs_from_spectrum(spectrum, feh=...)` for the
complete temperature workflow. Direct arrays use the same explicit declarations:

```python
from sc_teff_method import fit_hfs_spectrum, write_hfs_outputs
from sc_teff_method.hfs_synthetic import make_hfs_synthetic_spectrum

toy = make_hfs_synthetic_spectrum("noise")
analysis = fit_hfs_spectrum(
    toy.wavelength_air_angstrom, toy.flux,
    wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
    uncertainty=toy.uncertainty, mask=toy.mask, rv_kms=toy.rv_kms,
    configuration="nandakumar_feh", feh=-0.2,
)
for line, fit in analysis.fits.items():
    print(line, fit.status, fit.area_angstrom, fit.area_error_angstrom, fit.flags)
if analysis.temperature is not None:
    print(analysis.temperature.temperature_k, analysis.included_line_ids)
    print(analysis.performance.n_stars, analysis.performance.bias_k, analysis.performance.rmse_k)
write_hfs_outputs(analysis, "synthetic-hfs-output", spectrum_label="Analytical HFS example")
```

For areas measured separately, call `estimate_hfs_temperature` and
`hfs_subset_performance`. Both `nandakumar_feh` (explicit [Fe/H]) and the
independently fitted `nandakumar_no_feh` calibration are supported. The latter
never inserts an assumed [Fe/H] into its prediction. Any non-empty subset is
supported; omitted lines remain missing. The combined temperature is the
unweighted arithmetic mean of the individual line temperatures.

HFS areas must use HFS-specific coefficients. The original `estimate_temperature`
and `subset_performance` APIs retain their Gaussian meaning. Method-labelled
`analysis.areas` protects direct cross-use, and HFS-specific result/output types
preserve the profile identity. A plain numerical dictionary has no detectable
measurement origin: choosing a separately named API is an explicit declaration
by the caller. The frozen model-ID strings are shared across methods;
`measurement_source="hfs"` is essential to the numerical identity.

## Gaussian-anchor validation applicability

The frozen comparison's `strict_strength_domain` cut is on independently
measured **Gaussian-anchor areas ≥0.08 Å**, not on HFS areas. Supply a separate
JSON mapping with `--gaussian-anchor-areas gaussian-anchors.json`, or the Python
`gaussian_anchor_areas` argument, only when those Gaussian measurements are
available. Do not copy the HFS areas into this mapping.

Applicability is true only when all included lines have passing Gaussian
anchors, false when a known anchor fails, and otherwise unknown when required
anchors are missing. Missing anchors produce explicit warnings. HFS empirical
training ranges and reference-temperature/metallicity coverage are checked
separately and do not establish the Gaussian-anchor cut.

The frozen lookup retains all 511 subsets for each configuration, scope and
cohort. The workflow uses the included exact subset; the API also provides
`common_all9`. `all_available` removes the Gaussian-anchor selection cut but
does not establish reliable target measurements or instrument compatibility.
The held-out sample size, bias and RMSE remain descriptive empirical statistics
on the Nandakumar reference scale. A returned row is not a claim of target
applicability, and frozen RMSE does not validate post-hoc selection or new
continuum choices. Details and original hashes are in
[HFS calibration provenance](hfs_calibration_provenance.md).

## Optional manual continuum preprocessing

The input should already be approximately continuum-normalised. Optional
per-line corrections reuse PR 4's `ContinuumCorrection` and `apply_continuum`.
Each fit starts from the original spectrum, divides working flux and supplied
uncertainty by the chosen constant/linear C once, then fits with continuum fixed
at unity. Masks, rest coordinates and RV metadata are preserved. No additional
continuum free parameters or automatic normalisation method are introduced.

The `--continuum-config` JSON maps line IDs to manual controls. For example,
these are invented constant and linear choices, not recommendations for a star:

```json
{
  "line1": {"enabled": true, "mode": "constant", "level": 1.02},
  "line2": {
    "enabled": true,
    "mode": "linear",
    "anchors": [[21809.169, 0.99], [21815.169, 1.03]]
  }
}
```

Use `--continuum-warning-deviation` to set the provisional `abs(C-1)` warning
threshold (default 0.1), and `--no-continuum-extrapolation-warning` to disable
the provisional anchor-extrapolation warning. Finite positive continuum is
always required throughout fitted windows. Corrections are fully user-chosen
and do not reproduce the paper's continuum procedures. Their placement
uncertainty is excluded from formal covariance. HFS browser editing and review
states are not added in this PR; existing saved Gaussian reviews remain Gaussian.

## Outputs and scientific limits

`summary.json` records the HFS profile/area definition, unweighted procedure,
atomic dataset/policy/source hashes, calibration identity, source checksum and
parser/RV conventions, controls, masks, areas/errors/flags, line temperatures,
included subset and validation applicability. It embeds no input or corrected
spectral arrays. `lines.csv` retains all nine measurements and missing outcomes.
PNG/PDF figures show the actual corrected fit pixels, total HFS mixture,
component positions, rest/fitted parents, window bounds, exclusions and
corrected residuals. Use `write_hfs_outputs` or `plot_hfs_diagnostics`; Gaussian
plot/output functions reject HFS result objects. Existing output directories
need explicit overwrite permission, and source/configuration paths are protected.

Blends can bias the fitted common amplitude, parent centre and width. Continuum
placement directly changes areas. The fixed-strength Gaussian mixture does not
model saturation or full radiative transfer, and instrumental resolving power
can change profile compatibility. Good numerical fits and laboratory splitting
agreement do not establish reliable temperatures for arbitrary observations.
The empirical calibration remains tied to its measurement definition and
observational conditions. Inspect every contributing profile and assess those
conditions before scientific use.

The [verification guide](hfs_verification.md) records analytical tests and
reference comparisons. No observed IGRINS spectrum, extract, reviewed stellar
HFS fit or catalogue is required. FITS, interactive HFS review, stellar-atmosphere
synthesis, HFS-constant optimisation, automatic continuum finding, rectangular
bands and calibration refitting remain outside scope.
