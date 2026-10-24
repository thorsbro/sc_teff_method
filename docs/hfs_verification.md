# HFS numerical verification and development audit

The production identity is `hala2022_ab_generated` under atomic policy
`hfs-atomic-v2-2026-10-02`. The stellar calibration remains
`2026-10-05-nine-line-v1`, selected with `measurement_source="hfs"`.
No stellar calibration was refitted. The shared frozen model IDs do not by
themselves distinguish HFS from Gaussian measurements.

## Reproduction checks

The allowed reference `src/scteff/hfs.py` has SHA-256
`245ca2da19e0d0cbc524e762a5722b56b8e223be972069643b55635a075d16e8`
and Git blob SHA-1 `e6d7459d01e8c9b938fe8ba808a99ef6e3f69155`.
The source hash is checked before execution. The verifier executes those bytes
in an isolated module in memory, without importing the private package or
loading its spectrum/window/catalogue modules.

```bash
# Optional development-only comparison: exactly three permitted files are read.
python -B tools/verify_hfs_atomic_reference.py /path/to/read-only/reference
```

This command reads only the hash-checked standalone `hfs.py`, primary component
CSV and level CSV. It generates nine new analytical mixtures and independent
noise with seed 20261005. Maximum absolute differences were **0.0** for level
energy shifts, generated component offsets/Wigner strengths, all production
strengths, exact wavelength offsets, profile flux, fitted amplitude/centre/sigma,
full area and formal area error. A second independent nine-profile comparison
also found zero differences in these fitted quantities and residual RMS.

The committed `tests/fixtures/hfs_analytical_reference.json` contains scalar
inputs/results from a separate invented nine-profile comparison. It embeds no
spectral arrays and permits regression tests without private access. Portable
fit tolerances are 2e-8 for parameters/areas, and 2e-6 relative or 1e-10 absolute
for formal area errors. These cover small numerical-library differences; they
are not observational accuracy estimates. The clean analytical nine-line
generator recovers known full areas within 1.3e-10 Å with the reference solver's
default convergence tolerances.

The calibration extraction independently checks all **18 HFS coefficient rows**
and **4,088 subset rows** against the frozen aggregate files. Values agree
exactly as binary64 numbers; original validation strings are retained in the
public CSV. Across all 1,022 configuration/subset means, an independent
`math.fsum` calculation differs by at most 4.55e-13 K; scatter differences are
zero. Deterministic re-export reproduces all three calibration files byte for
byte. See [calibration provenance](hfs_calibration_provenance.md) and
[atomic provenance](hfs_atomic_data.md) for hashes and projection definitions.

## Independent scientific review

An independent subagent audited the 17 permitted source-file hashes and
rechecked the 133 component assignments/offsets/normalised strengths, 18
coefficient rows and 4,088 validation rows. No scientific blocker was found.

Its independent coupling checks cover 80 integer/half-integer transitions.
Strength marginals agree within 1.7e-16, wavenumber centroids within
8.5e-17 cm⁻¹ and transition-reversal strengths within 1.2e-16. Independent
integrated area/centroid/variance checks agree within 8.0e-17 on the tested
analytical profile. Finite-difference covariance agrees within 2.2e-8 relative
error, and the propagated area error within 1.3e-10 relative error. These checks
retain the amplitude–width cross term and distinguish the parent centre and
component sigma from the mixture's centroid and overall width.

Tests additionally cover both RV signs, masks, incomplete/gapped coverage,
parameter bounds, component centres near/crossing window edges, failed fits,
explicit quality policies, once-only manual continuum correction and error
scaling. The strict selection tests use separately supplied **Gaussian**
anchors; HFS area is never substituted for that anchor. Unknown applicability
is reported explicitly. Existing Gaussian regressions and saved-review fixtures
remain part of the complete installed-wheel suite.

## Reference-access audit

No real IGRINS spectrum, spectral extract, observed plot, stellar fitting
catalogue, individual-star measurement or observational template was read or
used. Intentional private reference reads used the requested allowlist; the
import-loader caveat below also concerns uninspected bytecode caches.
No manuscript, Data Availability Statement, private source/configuration or
frozen calibration product was edited. All permitted source/configuration hashes
remain unchanged.

Two development boundary issues are disclosed rather than claiming that the
whole private checkout was untouched:

- Early standalone imports used Python's source-file loader without disabling
  bytecode caching. They may have created or refreshed a private `__pycache__`
  entry. Those unlisted cache paths were not inspected or deleted. The final
  verifier uses in-memory compilation; a temporary-reference regression checks
  that it creates no bytecode/cache files in the reference directory.
- An initial review-agent search for `AGENTS.md` traversed sibling directory
  entries. It did not open non-allowlisted private file contents. Subsequent
  reference reads used exact allowed paths.

The extraction tools now reject output destinations inside the reference tree,
including aliases through symlinks and existing hard links. These checks use
temporary artificial reference trees in tests. No complete private repository,
Git history or complete publisher Table 4 is copied into the public package.

Analytical validation establishes numerical reproduction and software
correctness. Blends, saturation, instrumental resolution, continuum placement
and compatibility with the original measurement procedure remain scientific
limitations; no new observational accuracy or total temperature uncertainty is
claimed.
