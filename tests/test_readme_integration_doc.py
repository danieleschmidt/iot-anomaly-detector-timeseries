from pathlib import Path


def test_readme_mentions_integration_usage():
    """README should document how to run integration tests."""
    readme = Path("README.md").read_text()
    assert "pytest -m integration" in readme
    assert "make integration" in readme
