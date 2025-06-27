import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from src.visualize import main, plot_sequences


def test_plot_sequences(tmp_path):
    csv = 'data/raw/sensor_data.csv'
    out = tmp_path / 'plot.png'
    plot_sequences(csv, output=str(out))
    assert out.is_file()


def test_cli_creates_image(tmp_path):
    out = tmp_path / 'out.png'
    main(csv_path='data/raw/sensor_data.csv', output=str(out))
    assert out.is_file()
