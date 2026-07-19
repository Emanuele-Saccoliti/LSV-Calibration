import json
from pathlib import Path

import yaml
from lsv.cli import _format_price_report, run_engine


def test_single_command_pipeline_generates_report_and_3d_plot(
    tmp_path: Path,
) -> None:
    source_config = Path(__file__).parents[2] / "configs" / "calibration.yaml"
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    raw["experiment"]["output_directory"] = str(tmp_path)
    raw["experiment"]["show_plots"] = False
    config_path = tmp_path / "calibration.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    artifacts = run_engine(config_path, show_plots=False, quick=True, skip_exotic=True)

    assert artifacts.summary_path.is_file()
    assert artifacts.surface_data_path.is_file()
    assert artifacts.figure_path.is_file()
    assert artifacts.figure_path.stat().st_size > 10_000
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["quote_source"] == "synthetic"
    assert summary["static_arbitrage"]["is_arbitrage_free"]
    assert summary["heston"]["objective"] < 1e-10
    assert len(summary["heston"]["vanilla_prices"]) == 15
    assert len(artifacts.vanilla_prices) == 15
    report = _format_price_report(artifacts)
    assert "Vanilla prices (discounted calls)" in report
    assert "European call LSV" in report
    assert "DNT price:" in report
    assert summary["dnt"]["estimate"] > 0.0
