import json
from pathlib import Path
from typing import Optional

import lambench
from lambench.metrics.vishelper.results_fetcher import ResultsFetcher
from lambench.metrics.vishelper.metrics_calculations import MetricsCalculator
from lambench.metrics.vishelper.plot_generation import PlotGeneration
from datetime import datetime


class LAMBenchMetrics:
    def __init__(self, timestamp: Optional[datetime] = None):
        self.fetcher = ResultsFetcher(timestamp)
        self.metrics_calculations = MetricsCalculator(self.fetcher)
        self.plot_generation = PlotGeneration(self.fetcher, self.metrics_calculations)

    def save_results(self):
        raw_results = self.fetcher.aggregate_ood_results()
        barplot_data = self.plot_generation.generate_barplot(raw_results)
        radar_chart_config = self.plot_generation.generate_radar_plot(barplot_data)
        scatter_plot_data = self.plot_generation.generate_scatter_plot()
        downstream_domainwise, final_ranking = (
            self.metrics_calculations.summarize_final_rankings()
        )
        pc_barplot_data = self.plot_generation.generate_barplot(downstream_domainwise)

        result_path = Path(lambench.__file__).parent / "metrics/results"
        with open(result_path / "radar.json", "w") as f:
            json.dump(radar_chart_config, f, indent=2)
            f.write("\n")
        with open(result_path / "scatter.json", "w") as f:
            json.dump(scatter_plot_data, f, indent=2)
            f.write("\n")
        with open(result_path / "barplot.json", "w") as f:
            json.dump(barplot_data, f, indent=2)
            f.write("\n")
        with open(result_path / "final_rankings.json", "w") as f:
            json.dump(final_ranking.to_dict(orient="records"), f, indent=2)
            f.write("\n")
        with open(result_path / "pc_barplot.json", "w") as f:
            json.dump(pc_barplot_data, f, indent=2)
            f.write("\n")
        print("All plots saved to metrics/results/")


def main(timestamp: Optional[datetime] = None):
    metrics = LAMBenchMetrics(timestamp)
    metrics.save_results()


if __name__ == "__main__":
    main()
