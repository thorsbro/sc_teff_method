# Sc I HFS atomic definition and numerical provenance

The public HFS model uses only **`hala2022_ab_generated`**, frozen under atomic
policy **`hfs-atomic-v2-2026-10-02`**. Its component positions use the adopted
magnetic-dipole A **and** electric-quadrupole B constants. No laboratory constant,
individual component strength or component width is fitted to a stellar spectrum.

## Adopted atomic information

The isotope is **Sc-45**, with nuclear spin **I = 7/2**. The atomic source is
[Hala & Nave (2022), ApJS 259, 17](https://doi.org/10.3847/1538-4365/ac3edc),
*Hyperfine Structure Constants of Sc I and Sc II with Fourier Transform
Spectroscopy*. The article is CC BY 4.0; this attribution covers the selected
third-party atomic information, separately from the project's MIT software
licence. See [DATA_NOTICE.md](../DATA_NOTICE.md).

The compact export contains ten primary electronic levels, nine canonical air
wavelengths/atomic centre-of-gravity wavenumbers, and 133 component assignments:

| Line | Air parent (Å) | Atomic COG (cm⁻¹) | Components |
| --- | ---: | ---: | ---: |
| line1 | 21730.467 | 4600.57 | 21 |
| line2 | 21812.169 | 4583.37 | 16 |
| line3 | 21842.781 | 4576.91 | 9 |
| line4 | 22024.168 | 4539.21 | 12 |
| line5 | 22051.984 | 4533.47 | 21 |
| line6 | 22065.302 | 4530.78 | 18 |
| line7 | 22266.719 | 4489.78 | 6 |
| line8 | 22394.703 | 4464.13 | 12 |
| line9 | 22656.776 | 4412.45 | 18 |

The lower-level constants retain Hala & Nave's recommendations from Ertmer &
Hofer (1976) and Zeiske et al. (1976). All stored A/B constants and laboratory
uncertainties use cm⁻¹. Laboratory uncertainties are provenance, not additional
stellar fitting parameters or a propagated temperature uncertainty.

For the upper `4D1/2` level, B = 0 because J < 1 makes the quadrupole interaction
physically inapplicable. For upper `4D3/2`, `4D5/2` and `2D3/2`, Hala & Nave do
not report B: the production model explicitly assumes B = 0. These cases carry
`unreported_assumed_zero`, distinct from `not_applicable_J_lt_1`. Missing B is
not described as a measured zero.

Line9 retains the **positive-A** upper `2D5/2` assignment:
**A = +0.02205 cm⁻¹, B = −0.01581 cm⁻¹**. No historical negative-A alternative
is selected or exported.

## Positions, selection rules and relative strengths

For each level, F runs from |I−J| to I+J in unit steps. Electric-dipole
transitions satisfy ΔF = 0, ±1, excluding 0→0 and any zero Wigner strength;
electronic ΔJ and parity are also checked. With
`K = F(F+1) − I(I+1) − J(J+1)`, the first-order energy is

```text
E(F) = A K/2 + B [3 K(K+1)/4 − I(I+1) J(J+1)]
                    / [2 I(2I−1) J(2J−1)].
```

The B term is omitted when I < 1 or J < 1. A component offset is its upper
shift minus its lower shift. The supplied positions preserve the frozen
twelve-decimal A+B calculation; regeneration from the recorded constants is
checked to 5.01e-13 cm⁻¹.

The analytical Wigner 6-j strengths use an exact rational Racah sum squared.
They verify the allowed F assignments and the published rounding. Production
does **not** replace the published pattern with these exact Wigner values.

Hala & Nave Table 4 prints strengths to six decimals. Those raw values sum to
1.000001 for line1, 0.999999 for line5, and 1.000000 for the other seven lines.
The private frozen production table stores twelve-decimal normalised versions
of those rounded values. Public fitting loads those same twelve-decimal values
and performs the reference loader's final per-line normalisation, including
floating-point summation correction. The resulting Python sum is exactly one.
Both raw published rounded strengths and frozen production strengths are
recorded in the compact atomic file so the definitions remain inspectable.

## Table 4 is an A-only comparison here

For these nine transitions, the selected published Table 4 positions agree with
an **A-only** calculation to **2.36e-5 cm⁻¹**. With the adopted A+B constants,
the largest difference from Table 4 is **0.008399767857 cm⁻¹**, on line9. The
article's table note does not explicitly say B was omitted; the A-only finding
is a numerical property of this selected nine-transition corpus.

The public production pattern therefore uses **A+B positions**. The 133 selected
Table 4 positions are retained only as comparison metadata, not as a selectable
fitting pattern. The full publisher machine-readable table is neither read by
the export tool nor distributed. Its frozen source SHA-256 record is
`54a82bd6dfcdfed9b850a4a4ab60a3e1b8976a7b870266142db4dc3971c0dcb1`.

## Exact conversion to relative air-wavelength offsets

The canonical air parent λ₀ and atomic COG wavenumber ν₀ are independent inputs.
The implementation evaluates the exact relation in its stable form:

```text
Δλ = −λ₀ Δν / (ν₀ + Δν).
```

It neither subtracts two large nearby wavelengths nor approximates the relation
by `−λ₀ Δν/ν₀`. In particular, ν₀ is **not** calculated as `1e8/λ_air`. Zero
wavenumber offset stays exactly zero; positive wavenumber shift gives negative
wavelength shift. A fitted parent displacement moves every component together
without changing these frozen relative offsets.

## Verification and reproducibility

The installed package reads only its hash-checked files under
`data/hfs-atomic-v2-2026-10-02/`. `atomic_provenance.json` records SHA-256 and Git
blob identities for the explicitly permitted source files, the original
publisher source hash, projection rules and the derived atomic-file hash. The
manifest itself has a fixed SHA-256 anchor in the public loader. Legacy
comparison/sensitivity constants, approximate wavelength-offset columns,
private catalogues and all observed spectra are omitted.

`tools/export_hfs_atomic.py REFERENCE_ROOT DESTINATION` repeats the projection
using an explicit filename allowlist, without directory searches or private
package imports. `tools/verify_hfs_atomic_reference.py REFERENCE_ROOT` verifies
the standalone numerical reference checksum before importing only that file.
It compares newly generated analytical profiles, with seed **20261005**, never
observational spectra or observed templates.

On those artificial nine-line profiles the public and frozen standalone
reference agree with maximum absolute difference **0.0** for level shifts,
generated component offsets/strengths, production-strength normalisation,
wavelength offsets, profile fluxes, fitted amplitude/centre/width, full area and
formal area error. Public regression tests need no private repository. They
check all 133 assignments, energy/selection rules, known Wigner strengths,
high-precision Decimal wavelength conversion, Table 4/A+B distinctions,
positive line9 A, missing-B statuses, full-area integration and covariance.

The empirical profile is a fixed-strength Gaussian mixture, not radiative
transfer or stellar-atmosphere synthesis. It has no saturation parameter or
free isotope mixture. Blends, continuum placement, resolving power and
measurement compatibility require scientific assessment. Formal covariance
holds the atomic pattern and continuum fixed; it excludes those systematics.
Analytical recovery establishes numerical correctness, not observational
accuracy or transfer accuracy for another instrument. Further atomic-data or
co-author permissions must be confirmed if scope expands beyond this attributed
selection; no broader permission is asserted.
