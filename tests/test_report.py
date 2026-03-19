"""Tests for AnomalyReport and AnomalyFinding."""

import pytest
from iot_anomaly_detector.report import AnomalyFinding, AnomalyReport, Severity


def make_finding(t=10, sensor="temperature", severity=Severity.HIGH, confidence=0.9):
    return AnomalyFinding(
        timestep=t,
        timestamp=float(t * 60),
        sensor=sensor,
        severity=severity,
        anomaly_type="point",
        confidence=confidence,
        detector="test",
        score=5.0,
        value=42.0,
    )


class TestAnomalyReport:
    def test_empty_report(self):
        r = AnomalyReport()
        assert r.n_findings == 0

    def test_add_finding(self):
        r = AnomalyReport()
        r.add(make_finding())
        assert r.n_findings == 1

    def test_merge(self):
        r1 = AnomalyReport([make_finding(t=5)])
        r2 = AnomalyReport([make_finding(t=10), make_finding(t=15)])
        merged = r1.merge(r2)
        assert merged.n_findings == 3

    def test_detected_timesteps_unique_sorted(self):
        r = AnomalyReport([
            make_finding(t=10),
            make_finding(t=5),
            make_finding(t=10),  # duplicate
        ])
        ts = r.detected_timesteps()
        assert ts == [5, 10]

    def test_critical_count(self):
        r = AnomalyReport([
            make_finding(severity=Severity.CRITICAL),
            make_finding(severity=Severity.HIGH),
            make_finding(severity=Severity.CRITICAL),
        ])
        assert r.critical_count == 2
        assert r.high_count == 1

    def test_by_sensor(self):
        r = AnomalyReport([
            make_finding(sensor="temperature"),
            make_finding(sensor="pressure"),
            make_finding(sensor="temperature"),
        ])
        by_s = r.by_sensor()
        assert len(by_s["temperature"]) == 2
        assert len(by_s["pressure"]) == 1

    def test_summary_str(self):
        r = AnomalyReport([make_finding(), make_finding(t=20, sensor="pressure")])
        summary = r.summary()
        assert "2 findings" in summary
        assert "temperature" in summary
        assert "pressure" in summary

    def test_repr(self):
        r = AnomalyReport([make_finding()])
        assert "n=1" in repr(r)
