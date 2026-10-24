"""Method-labelled, immutable line areas without numerical dependencies."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class LineAreas(Mapping[str, float]):
    """Areas carrying their measurement definition across calibration calls.

    Plain numerical dictionaries cannot reveal how an area was measured. The
    spectral workflows use this mapping to detect accidental method mixing;
    the corresponding estimator performs numerical and line-ID validation.
    """

    areas: Mapping[str, float]
    measurement_source: str

    def __post_init__(self):
        if self.measurement_source not in ("gaussian", "hfs"):
            raise ValueError("measurement_source must be gaussian or hfs")
        if not isinstance(self.areas, Mapping):
            raise TypeError("areas must be a mapping of line areas")
        object.__setattr__(self, "areas", MappingProxyType(dict(self.areas)))

    def __getitem__(self, key: str) -> float:
        return self.areas[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.areas)

    def __len__(self) -> int:
        return len(self.areas)
