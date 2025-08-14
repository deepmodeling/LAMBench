import numpy as np


class PlotGeneration:
    def __init__(self, fetcher, metrics_calculator):
        self.fetcher = fetcher
        self.metrics_calculator = metrics_calculator

    def generate_radar_plot(self, barplot_data: dict) -> dict:
        categories = list(barplot_data.keys())
        models = list(barplot_data[categories[0]].keys())

        radar_values = {}
        for domain, domain_results in barplot_data.items():
            for model, value in domain_results.items():
                if model not in radar_values:
                    radar_values[model] = [None] * len(categories)
                radar_values[model][categories.index(domain)] = (
                    1 - value if value is not None else None
                )
        best_model = max(
            radar_values,
            key=lambda k: np.sum(radar_values[k]) if None not in radar_values[k] else 0,
        )
        return self._build_radar_chart_config(
            categories, radar_values, models, best_model
        )

    def generate_scatter_plot(self) -> list[dict]:
        results = []
        for model in self.fetcher.leaderboard_models:
            efficiency_raw = (
                self.fetcher.fetch_inference_efficiency_results_for_one_model(model)
            )
            zeroshot_raw = self.metrics_calculator.calculate_mean_m_bar_domain(model)
            if (
                efficiency_raw is None
                or efficiency_raw["average_time"] is None
                or zeroshot_raw is None
            ):
                continue
            results.append(
                {
                    "name": model.model_metadata.pretty_name,
                    "family": model.model_family,
                    "nparams": model.model_metadata.num_parameters,
                    "efficiency": np.round(efficiency_raw["average_time"], 2),
                    "std": np.round(efficiency_raw["standard_deviation"], 2),
                    "generalizability error": np.round(zeroshot_raw, 2),
                }
            )
        return results

    def generate_barplot(self, domain_results: dict) -> dict:
        results = {}
        for model, domain_result in domain_results.items():
            for domain, metrics in domain_result.items():
                if domain not in results:
                    results[domain] = {}
                if metrics is not None:
                    results[domain][model] = np.round(metrics, 2)
                else:
                    results[domain][model] = None
        return results

    def _build_radar_chart_config(
        self,
        categories: list[str],
        radar_values: dict[str, list[float | None]],
        models: list[str],
        best_model: str | None,
        text_color: str = "white",
    ) -> dict:
        area_style = {"areaStyle": {"opacity": 0.1}}
        chart_config = {
            "legend": {"data": models, "bottom": 0, "textStyle": {"color": text_color}},
            "radar": {
                "indicator": [{"name": category, "max": 1, "min": 0.5} for category in categories],
                "axisName": {"color": text_color},
            },
            "series": [
                {
                    "name": "LAMBench Leaderboard",
                    "type": "radar",
                    "data": [
                        {"name": model, "value": values}
                        for model, values in radar_values.items()
                    ],
                }
            ],
        }
        if best_model:
            for model_data in chart_config["series"][0]["data"]:
                if model_data["name"] == best_model:
                    model_data.update(area_style)
        return chart_config
