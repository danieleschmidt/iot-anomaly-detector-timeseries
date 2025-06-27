import subprocess
import sys
from pathlib import Path
import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("tensorflow")


ROOT = Path(__file__).resolve().parents[1]


def test_integration_via_marker(tmp_path):
    """Verify the integration test can be run via pytest using the integration marker."""
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-m', 'integration', str(ROOT / 'tests' / 'test_integration_pipeline.py'), '-q'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    assert result.returncode == 0
    assert 'passed' in result.stdout

