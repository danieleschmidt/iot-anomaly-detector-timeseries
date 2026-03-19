"""Structured anomaly report — findings with metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyFinding:
    """A single detected anomaly."""
    timestep: int
    timestamp: float
    sensor: str
    severity: Severity
    anomaly_type: str        # "point", "contextual", "collective", "unknown"
    confidence: float        # 0.0 – 1.0
    detector: str            # Which detector produced this
    score: float             # Raw anomaly score (detector-specific)
    value: Optional[float] = None     # Sensor value at detection
    description: str = ""


@dataclass
class AnomalyReport:
    """
    Aggregated anomaly findings from one or more detectors.

    Supports merging findings from multiple detectors and computing
    simple summary statistics.
    """
    findings: List[AnomalyFinding] = field(default_factory=list)

    def add(self, finding: AnomalyFinding) -> None:
        self.findings.append(finding)

    def merge(self, other: "AnomalyReport") -> "AnomalyReport":
        """Return a new report combining both sets of findings."""
        return AnomalyReport(findings=self.findings + other.findings)

    @property
    def n_findings(self) -> int:
        return len(self.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    def by_sensor(self) -> dict:
        result: dict = {}
        for f in self.findings:
            result.setdefault(f.sensor, []).append(f)
        return result

    def detected_timesteps(self) -> List[int]:
        """Return sorted list of all timesteps flagged."""
        return sorted({f.timestep for f in self.findings})

    def summary(self) -> str:
        lines = [
            f"AnomalyReport: {self.n_findings} findings",
            f"  Critical: {self.critical_count}  High: {self.high_count}",
        ]
        by_s = self.by_sensor()
        for sensor, fs in sorted(by_s.items()):
            lines.append(f"  {sensor}: {len(fs)} anomalies")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AnomalyReport(n={self.n_findings})"
