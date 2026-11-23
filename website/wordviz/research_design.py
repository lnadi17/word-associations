"""Directional target sets and their empirical provenance."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PairedStudy:
    primes: tuple[str, str]
    labels: tuple[str, str]
    targets: tuple[tuple[str, ...], tuple[str, ...]]
    citation: str
    url: str
    context: str
    data_url: str = ""
    evidence_url: str = ""
