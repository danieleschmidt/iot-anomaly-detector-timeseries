from pathlib import Path


def test_readme_mentions_dev_setup():
    readme = Path("README.md").read_text()
    assert "pip install -r requirements-dev.txt" in readme
    assert "scripts/test.sh" in readme

