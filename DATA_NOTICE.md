# Software and calibration products

The [MIT licence](LICENSE) covers the software, including the analytical example
generators and tests. Calibration products and selected third-party atomic
information are described separately below.

## Frozen empirical calibration products

The package contains a limited projection of the frozen
`2026-10-05-nine-line-v1` Gaussian and HFS calibration products:
two configurations per method (with and without [Fe/H]), each with coefficients
for all nine Sc I lines. The products also include
marginal empirical coverage and aggregate fixed-subset validation sample sizes,
biases and RMSE values. Gaussian and HFS products have separate hash-checked
package data and provenance. See [Gaussian provenance](docs/provenance.md) and
[HFS calibration provenance](docs/hfs_calibration_provenance.md).

These numerical calibration products were derived from the collaboration's
scientific research and are publicly available in this repository. With the
agreement of the co-authors, the released Gaussian and HFS calibration
coefficients, empirical coverage summaries, aggregate fixed-subset validation
statistics, and associated numerical-data provenance are licensed under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Users may share and adapt these released calibration products under CC BY 4.0,
provided they credit the Sc I calibration authors, identify this repository as
the source, link to the licence, and indicate any changes. Please also cite the
accompanying scientific paper once its final citation becomes available.
The CC BY 4.0 grant for these calibration products does not extend to the
underlying observational spectra or other third-party observational data.
The software remains licensed separately under MIT.

## Selected third-party HFS atomic data

The HFS atomic information is adapted from Hala & Nave (2022),
*Hyperfine Structure Constants of Sc I and Sc II with Fourier Transform
Spectroscopy*, **ApJS 259, 17**, DOI
[10.3847/1538-4365/ac3edc](https://doi.org/10.3847/1538-4365/ac3edc).
The article identifies its publication under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/);
the attribution and licence statement are also present in the
[author-hosted article](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=929570).
This third-party attribution and licence are independent of both the MIT
software licence and the CC BY 4.0 licence granted for the empirical calibration
products above.

Only the nine-transition selection needed by this method is included: ten
primary electronic levels and 133 component assignments. The full publisher
machine-readable Table 4 is not distributed. The lower-level constants retain
Hala & Nave's recommended earlier measurements. Source checksums and specific
provenance are retained in the packaged `atomic_provenance.json` and described
in [the atomic-data guide](docs/hfs_atomic_data.md).

Adaptations comprise generation of first-order A+B component positions from
the adopted constants, retention of the published F assignments, normalisation
of the published rounded strengths as in the frozen reference, and conversion
to relative offsets around the public canonical air wavelengths. The selected
published Table 4 positions are comparison metadata, not the production A+B
pattern. Missing-B assumptions are explicit; no fitted stellar constants or
individual-star HFS measurements are distributed. Permissions for additional
datasets or expanded source selections must be checked separately.

## Spectral inputs and examples

Users supply their own appropriately prepared spectra when applying the fitting
tools. The original calibration spectra are not publicly distributed. The
example spectra and diagnostic figures provided with this repository are
synthetic, based on Gaussian and HFS profiles with reproducible noise where
applicable. Permissions to redistribute underlying observations are separate
from the software and calibration-data licences described above.
