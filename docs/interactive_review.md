# Local interactive line review

Install `python -m pip install ".[inspect]"` from a checkout. Bokeh is used for
Python callbacks and interactive plots, with assets served locally rather than
by a CDN. The measured-area base install and `spectra` extra remain usable without
Bokeh. Python 3.11+ is retained with compatible Bokeh 3.x (`>=3.6,<4`).

## Start with analytical data

```bash
python examples/demo_interactive_review.py --output-dir review-demo
sc-teff inspect review-demo/spectrum.csv \
  --delimiter , --header --wavelength-column wavelength --flux-column flux \
  --uncertainty-column uncertainty --mask-column exclude \
  --wavelength-unit angstrom --wavelength-frame air --continuum-normalised \
  --feh -0.2 --save-review review.json --output-dir reviewed-results \
  --label "Analytical synthetic review"
```

The server binds only `127.0.0.1`, chooses a free port and opens your browser.
`--port` chooses a local port; `--no-browser` prints the URL for manual opening.
Save before closing/reloading a tab. Ctrl-C stops the server. Tabs have independent
unsaved decisions; a saved/loaded review becomes the starting point for a new tab.
There are no cloud services, telemetry, remote uploads or external asset requests,
including when a user's Bokeh settings normally request CDN resources.

For your own data, use PR 2's column, delimiter, uncertainty, pixel-mask,
window/initial and `--fit-config` options. Explicitly declare continuum-normalised
air wavelengths in Å. No automatic continuum normalisation, unit/vacuum conversion or RV
inference occurs. Positive `--rv-kms` means recession; PR 2 applies the relativistic
rest correction once. All displayed windows, centres and masks use corrected
rest air Å. The header records calibration, [Fe/H] and RV provenance.

For no metallicity input, choose `--configuration nandakumar_no_feh`. Calibration
and input conventions/global options are fixed for the session. They can also be
restored from a same-input review with the chosen configuration.

## Refine one line

The overview shows all nine profiles and residuals, laboratory/fitted centres,
window boundaries, excluded pixels, areas, diagnostic Teff and numerical flags.
Click a line or use **Detailed line** for a larger view; navigation retains edits.

1. Edit **Window lower/upper**, then Enter or Tab. The existing PR 2 fitter refits
   only this line; the other eight measurements stay cached.
2. Drag horizontally in the detailed profile to add a mask. Select an interval
   to edit its numerical endpoints, update it or remove it. Original input-file
   pixel masks remain in force alongside added intervals.
3. Uncheck **Automatic Gaussian initial guess** to edit depth, centre and sigma.
   Sigma is a Gaussian standard deviation in Å, not FWHM.
4. **Reset line defaults** restores the default window, automatic start and no
   added intervals, and resets local continuum to unity. Input-file masks, RV
   and global fit options are retained.
   **Refit line** also requires renewed inspection.

Any edit invalidates that line's previous acceptance, even if the new controls
converge to the same fitted parameters. Invalid drafts remain visible across
navigation and block acceptance, saving and export until repaired/reset. A last
valid fit may remain visible as a diagnostic, with pending/unapplied status.

## Adjust the local continuum manually

The source must already be approximately continuum-normalised. This optional
correction is a user decision, never a continuum estimate inferred from pixels.
It is not a reproduction of the paper's continuum-normalisation procedures.

- **No correction:** C = 1, preserving the existing numerical behaviour.
- **Constant:** enter one finite, positive continuum level.
- **Linear:** specify two different rest-air-Å wavelengths and their positive
  continuum values. The straight line interpolates between them and extrapolates
  outside them; extrapolation prompts a warning.

Enable local continuum adjustment in the detailed view. Use the numerical
fields for precision, or choose continuum-anchor mouse interaction and place
the two points in the profile. Move/remove anchors using the provided controls;
return to mask interaction to exclude a blend. No peak detection, optimisation
or polynomial fit chooses these values for you. Disabling the correction keeps
the controls but uses unity; resetting continuum discards them. A valid change
immediately refits only the selected line. Incomplete/invalid drafts block
acceptance and saving until corrected or reset, as with fitting-window drafts.

Every refit starts from the immutable original spectrum and applies
`F_corrected = F_input / C` exactly once. Supplied absolute pixel errors also
become `sigma_corrected = sigma_input / C`. Wavelengths, RV metadata and masks
are preserved. The bounded Gaussian fit still has exactly amplitude, centre
and width as free parameters, with its continuum fixed at unity. Continuum
anchors are not Gaussian optimisation parameters.

The detailed plots distinguish original input, corrected fit pixels, C, and
the Gaussian model. Residuals use corrected flux. Area and individual Teff are
shown both before and after correction for the current window/masks/initial
controls. Neither diagnostic comparison approves a fit. Any change, including
disabling/resetting a correction or changing an inactive control, invalidates
that line's prior acceptance. Inspect and explicitly accept it again.

Continuum must stay finite and strictly positive throughout the fitting window,
including masked pixels and endpoints. Invalid corrections are refused. The
provisional `continuum_large_correction` warning uses `abs(C-1) > 0.1`;
`continuum_anchor_extrapolation` warns when linear anchors do not bracket the
whole window. These are inspection prompts, not acceptance tests. For a new
session configure `--continuum-warning-deviation 0.05` or
`--no-continuum-extrapolation-warning`; the choices are saved and restored.
Python callers can supply `ContinuumOptions` to `ReviewSession.create`.

```python
from sc_teff_method import ContinuumCorrection

# session is an existing ReviewSession over your original prepared Spectrum.
session.edit_line("line2", continuum=ContinuumCorrection(
    enabled=True, mode="constant", level=1.035,
))
# Inspect the revised fit before deciding, even if the fit converged.
session.decide("line2", "accepted", "Inspected the chosen manual correction.")
```

**Continuum placement influences areas and may introduce systematic differences
from the empirical calibration's measurement definition.** The user is
responsible for evaluating the continuum choices. Formal Gaussian covariance
holds those choices fixed and excludes continuum-placement uncertainty. No
unvalidated total temperature error is added.

## Accept or reject scientifically

Every line begins Pending unless explicitly excluded at launch. **Accept fit**
includes a valid measurement; **Accept flagged fit** makes deliberate inclusion
explicit while preserving flags. A short reason is optional. Failed, unavailable
or numerically excluded lines cannot be accepted. **Reject** omits a measurement;
**Pending** withdraws a decision. Convergence never approves a fit automatically.

Only accepted valid positive areas enter PR 1's unweighted mean of line
temperatures. No accepted lines leaves reviewed temperature and performance
missing. Diagnostic per-line temperatures for pending/rejected fits are labelled
separately by their review state and do not contribute to the final estimate.

The header updates Teff, contributing identity, sample line scatter, exact-subset
N/bias/RMSE and applicability warnings. These statistics use the frozen subset
lookup, not a new validation experiment. **Frozen subset RMSE does not validate
an arbitrary post-hoc line-selection procedure or newly chosen continuum
corrections.** Manual inspection can introduce
selection bias, and line scatter is not the total temperature uncertainty.

## Save, restore and replay

Choose a path and click **Save review JSON**. Existing files need the explicit
replacement checkbox. **Load same-spectrum review** restores after identity
checks. **Export reviewed results** writes a complete new output directory.

```bash
sc-teff inspect review-demo/spectrum.csv --review-file review.json
sc-teff replay review-demo/spectrum.csv --review-file review.json \
  --output-dir final-results --plot-format both
```

Replay needs the `spectra` extra, without Bokeh/browser. Saved input declarations,
parser settings and RV correction are used; do not reapply RV to the source or
mix new input/fitting overrides into restore. `fit --fit-config` is unchanged.
Replay returns 3 when no valid lines are accepted while still writing diagnostics;
invalid inputs/stale reviews return 2.

The human-readable `sc-teff-review` schema version 2 stores source and prepared
SHA-256 hashes, portable basename/parser settings, coordinate/RV conventions,
frozen calibration/configuration/[Fe/H], global options, all nine line controls,
decisions/reasons, notes, continuum modes/enabled states/levels/anchors,
continuum warning options, scalar numerical snapshots and warnings. It embeds no
spectrum. Identical copies can be moved/renamed; even whitespace-only source-byte
changes fail the checksum check.

Restore recalculates fits from the source. Saved scalar areas and temperatures
are audit records, not calculation inputs. Accepted decisions are bound to their
controls and compared with recomputed fits. Stale approvals require renewed
inspection. These checks detect stale reviews; the files contain user decisions,
not authenticated scientific approval. Schema version 1 reviews migrate to
identity continuum after their original acceptance fingerprints and numerical
snapshots are checked; their fitting and temperature results remain unchanged.
Saved areas are never substituted for a refit, in either schema.

Export includes `review.json`, `summary.json`, `lines.csv` and PNG/PDF diagnostics,
with all nine numerical and review statuses. Missing values remain null/empty.
JSON/CSV include continuum controls, matched-control uncorrected areas and line
temperatures, final formal area errors, flags and manual decisions. PNG/PDF
show the corrected pixels actually fitted, with original flux and continuum for
comparison. Formal area errors exclude continuum-placement uncertainty.
Existing directories need explicit overwrite; source/configuration/review inputs
are protected against export collisions. Deliberately saving an existing working
review is a separate operation. Original spectrum files and arrays are never edited.

## Complete synthetic exercise and verification

Add `--scripted-review` to the example-generator command for a completed
illustrative review and numerical comparison. Profiles, neighbours, noise
(seed 20261023), masks and coverage loss are invented. Line1 is clean; line2 has
an input mask; line3 has incomplete coverage; line4 has an invented neighbour;
line6 has imposed continuum errors; line8 has another blend. Several lines can
be accepted after inspection, preserving flags where deliberately accepted.

The scripted exercise accepts line4, masks 22024.318–22025.768 Å, confirms that
approval is invalidated and explicitly reaccepts it. Area changes 0.63244→0.50307 Å
against analytical truth 0.49902 Å. On the same accepted subset, Teff changes
3554.58→3567.27 K. These are numerical examples, not observed stellar temperatures.
The committed browser screenshot uses this entirely analytical demonstration.
No private repository or observational spectrum is needed or accessed.

```bash
python -m pip install -e ".[dev]"
python -m playwright install chromium
python -m pytest -q
```

Tests cover PR 2 fitter reuse, independent analytical parity, approval
invalidation, checksums, portability, replay, safe output and local server
startup/shutdown. Real Chromium tests exercise buttons, windows, mouse masks and
save/restore/export with external requests blocked. Browser tests skip if the
optional test browser is absent; normal inspection uses the user's own browser.

## Analytical continuum exercise

```bash
python examples/demo_continuum_review.py --output-dir continuum-demo --scripted-review
sc-teff inspect continuum-demo/spectrum.csv --review-file continuum-demo/review.json
sc-teff replay continuum-demo/spectrum.csv --review-file continuum-demo/review.json \
  --output-dir continuum-final --plot-format both
```

This separate generator leaves earlier synthetic examples unchanged. It uses
only mathematical Gaussian profiles, invented levels/slopes/blends and a fixed
seed (20261024). Default noise is zero for comparison with analytical truth;
optional independently generated noise has a declared seed and standard
deviation. Line1 has unity continuum, line2 a constant offset, line3 a linear
slope, line4 a neighbouring blend, line5 deliberately wrong example anchors,
line7 an input mask and line8 incomplete coverage. The scripted review records
before/after comparisons and explicit synthetic decisions. Correct prescribed
continua recover the known Gaussian areas; wrong anchors alter area and Teff
without establishing a better measurement.

For example, the correct constant changes line2's area from 0.42892810 to
0.48856692 Å, matching analytical truth. Deliberately wrong line5 anchors change
its area from 0.54880119 to 0.62814265 Å and the same seven-line mean from
3512.834579 to 3503.553363 K after explicit reapproval. The saved demonstration
restores line5 to unity and accepts it again. The unblended synthetic cases
recover prescribed areas within 3.1e-13 Å and the analytical-area temperature
within 3.3e-11 K; these tolerances describe numerical correctness only.

![Analytical synthetic continuum review](images/continuum_review.png)

The preview is captured from the running local Bokeh application with this
analytical spectrum. Tests cover flux/error transformation, unity parity,
single-line isolation, repeated refits, acceptance invalidation, schema 1
migration, save/restore, headless replay, and browser anchor editing. CI runs
tests against an installed wheel outside the source checkout.

The Gaussian model stays fixed at unity after manual division. Formal parameter/area errors
exclude continuum placement,
profile and calibration systematics. Accuracy depends on observational and
measurement compatibility; synthetic tests establish software correctness, not
accuracy for arbitrary instruments. Scientific assessment of defaults, thresholds
and instrument transfer remains necessary. Automatic continuum finding, higher
polynomials, FITS, HFS, stellar synthesis and calibration refitting remain outside
scope. No private repository or observational data are used.
