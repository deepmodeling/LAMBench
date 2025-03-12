import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

import lambench
from lambench.databases.calculator_table import CalculatorRecord
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.metrics.post_process import DIRECT_TASK_WEIGHTS
from lambench.metrics.utils import (
    aggregated_inference_efficiency_results,
    aggregated_nve_md_results,
    exp_average,
    filter_direct_task_results,
    get_domain_to_direct_task_mapping,
    get_leaderboard_models,
)
from lambench.models.basemodel import BaseLargeAtomModel


def aggregate_domain_results_for_one_model(model: BaseLargeAtomModel):
    """
    This function aggregates the results for one model across domains.
    """

    domain_results = {}
    for domain, tasks in get_domain_to_direct_task_mapping(DIRECT_TASK_WEIGHTS).items():
        norm_log_results = []
        weight_virial = False
        for task in tasks:
            task_result = DirectPredictRecord.query(
                model_name=model.model_name, task_name=task
            )
            task_config = DIRECT_TASK_WEIGHTS[task]
            if task_config["virial_weight"] is not None:
                weight_virial = True
            if len(task_result) != 1:
                logging.warning(
                    f"Expect one record for {model.model_name} and {task}, but got {len(task_result)}"
                )
                continue

            norm_log_results.append(
                filter_direct_task_results(
                    task_result[0].to_dict(), task_config, normalize=True
                )
            )
        if len(norm_log_results) != len(tasks):
            domain_results[domain] = None
        else:
            domain_results[domain] = exp_average(norm_log_results)

            # aggregate over E, F, V, TODO refactor, need to fix normalization
            # currently std used in normalization is per atom based, so the normalization
            # only works for per atom based metrics
            normalized_e = domain_results[domain]["energy_mae_natoms"]
            normalized_f = domain_results[domain]["force_mae"]
            normalized_v = domain_results[domain]["virial_mae_natoms"]
            if weight_virial:
                domain_results[domain] = (
                    0.45 * normalized_e + 0.45 * normalized_f + 0.1 * normalized_v
                )
            else:
                domain_results[domain] = 0.5 * normalized_e + 0.5 * normalized_f
    return domain_results


def fetch_overall_zero_shot_results(
    model: BaseLargeAtomModel,
) -> float:
    """This function further aggregates the results for one model across all domains, used in scatterplot generation."""
    domain_results = list(aggregate_domain_results_for_one_model(model).values())
    return (
        np.mean(list(aggregate_domain_results_for_one_model(model).values()))
        if None not in domain_results
        else None
    )


def fetch_conservativeness_results(
    model: BaseLargeAtomModel,
    conservativeness_thresh: Optional[float] = 2e-5,
) -> float:
    """
    Fetch conservativeness results from NVE MD task for a given model.

    The conservativeness metric is calculated as (slope + std) / 2 - log(success_rate) / 100.
    - 'slope': energy drift slope in molecular dynamics simulation
    - 'std': standard deviation of energy during simulation
    - 'success_rate': percentage of successful simulation runs

    Lower values indicate better conservativeness, with penalties for failed simulations.

    Returns:
        float: Combined conservativeness metric, or None if results are not available
    """
    task_results = CalculatorRecord.query(
        model_name=model.model_name, task_name="nve_md"
    )

    if len(task_results) != 1:
        logging.warning(
            f"Expected one record for {model.model_name} and nve_md, but got {len(task_results)}"
        )
        return None

    metrics = aggregated_nve_md_results(task_results[0].metrics)
    slope = metrics["slope"]
    std = metrics["std"]
    success_rate = metrics["success_rate"]
    if conservativeness_thresh:
        return max(
            conservativeness_thresh, (slope + std) / 2 - np.log(success_rate) / 100
        )
    return (slope + std) / 2 - np.log(
        success_rate
    ) / 100  # to penalize failed simulations


def fetch_inference_efficiency_results(model: BaseLargeAtomModel) -> float:
    task_results = CalculatorRecord.query(
        model_name=model.model_name, task_name="inference_efficiency"
    )

    if len(task_results) != 1:
        logging.warning(
            f"Expected one record for {model.model_name} and inference_efficiency, but got {len(task_results)}"
        )
        return None

    metrics = aggregated_inference_efficiency_results(task_results[0].metrics)
    return metrics["average_time_per_step"]


def aggregate_domain_results() -> dict[str, dict[str, float]]:
    """
    This function aggregates the results across models and domains.
    """
    results = {}

    leaderboard_models = get_leaderboard_models()
    for model in leaderboard_models:
        domain_results = aggregate_domain_results_for_one_model(model)
        results[model.model_name] = domain_results

    return results


def generate_radar_plot(domain_results: dict) -> dict:
    """
    Generate radar plot data for domain results comparison.

    Args:
        domain_results: Dictionary mapping model names to their metrics across domains

    Returns:
        Dictionary containing the radar chart configuration for visualization
    """
    # Extract categories and models
    first_model = list(domain_results.keys())[0]
    categories = list(domain_results[first_model].keys())
    models = list(domain_results.keys())

    # Collect and process metrics
    metrics_data = _collect_metrics_data(domain_results, categories, models)
    normalized_metrics = _normalize_metrics(
        domain_results, metrics_data["category_max"], categories
    )
    model_rankings = _calculate_model_rankings(
        models, categories, metrics_data["category_values"]
    )
    best_model = _find_best_model(model_rankings["total_rankings"])

    # Generate the radar chart configuration
    return _build_radar_chart_config(categories, normalized_metrics, models, best_model)


def generate_scatter_plot() -> list[dict]:
    results = []
    leaderboard_models = get_leaderboard_models()
    for model in leaderboard_models:
        results.append(
            {
                "name": model.model_name,
                "family": model.model_family,
                "nparams": model.model_metadata.num_parameters,
                "efficiency": np.round(
                    1 / fetch_inference_efficiency_results(model), 2
                ),  # frames per second
                "zeroshot": np.round(
                    fetch_overall_zero_shot_results(model), 2
                ),  # unitless zero-shot metric across domains
            }
        )
    return results


def generate_barplot(domain_results: dict) -> dict:
    """Rearrange the domain results for barplot visualization"""
    results = {}
    for model, domain_result in domain_results.items():
        for domain, metrics in domain_result.items():
            if metrics is not None:
                if domain not in results:
                    results[domain] = {}
                results[domain][model] = np.round(metrics, 2)
    return results


def _collect_metrics_data(
    domain_results: dict[str, dict[str, float]],
    categories: list[str],
    models: list[str],
) -> dict[str, dict]:
    """Collect and process raw metrics data"""
    category_values: dict[str, list[float]] = {category: [] for category in categories}

    for model in models:
        for category in categories:
            if domain_results[model][category]:
                # Convert values to log scale (higher is better)
                category_values[category].append(
                    -np.log(domain_results[model][category])
                )

    # Find maximum for each category for normalization
    category_max: dict[str, float] = {
        category: max(values) if values else 1.0
        for category, values in category_values.items()
    }

    return {"category_values": category_values, "category_max": category_max}


def _normalize_metrics(
    domain_results: dict[str, dict[str, float]],
    category_max: dict[str, float],
    categories: list[str],
) -> dict[str, list[float | None]]:
    """Normalize metrics for each model across categories"""
    normalized_metrics: dict[str, list[float | None]] = {}

    for model, res in domain_results.items():
        normalized_metrics[model] = []
        for category in categories:
            if res[category]:
                # Normalize to [0,1] range
                normalized_value = (-np.log(res[category])) / category_max[category]
                normalized_metrics[model].append(normalized_value)
            else:
                normalized_metrics[model].append(None)

    return normalized_metrics


def _calculate_model_rankings(
    models: list[str], categories: list[str], category_values: dict[str, list[float]]
) -> dict[str, dict]:
    """Calculate rankings for models across categories"""
    category_rankings: dict[str, dict[str, int]] = {}

    for category in categories:
        category_rankings[category] = {}
        values = category_values[category]
        if values:  # Skip empty categories
            sorted_values = sorted(values, reverse=True)  # Higher is better
            model_values = {
                model: value
                for model, value in zip(models, values)
                if value is not None
            }
            for model, value in model_values.items():
                rank = sorted_values.index(value) + 1
                category_rankings[category][model] = rank

    # Calculate total rankings across all categories
    total_rankings: dict[str, int] = {}
    for model in models:
        total_rankings[model] = sum(
            category_rankings[category].get(model, 0) for category in categories
        )

    return {"category_rankings": category_rankings, "total_rankings": total_rankings}


def _find_best_model(total_rankings: dict[str, int]) -> str | None:
    """Find the model with the best overall ranking"""
    return min(total_rankings, key=total_rankings.get) if total_rankings else None


def _build_radar_chart_config(
    categories: list[str],
    normalized_metrics: dict[str, list[float | None]],
    models: list[str],
    best_model: str | None,
) -> dict:
    """Build the radar chart configuration"""
    # Define area style for the best model
    area_style: dict = {
        "areaStyle": {"opacity": 0.1},  # Use inherited color
    }

    # Build chart configuration
    chart_config: dict = {
        "title": {"text": "LAMBench Leaderboard"},
        "legend": {"data": models},
        "radar": {
            "indicator": [{"name": category, "max": 1} for category in categories]
        },
        "series": [
            {
                "name": "LAMBench Leaderboard",
                "type": "radar",
                "data": [
                    {"value": values, "name": model}
                    for model, values in normalized_metrics.items()
                ],
            }
        ],
    }

    # Highlight best model
    if best_model:
        for model_data in chart_config["series"][0]["data"]:
            if model_data["name"] == best_model:
                model_data.update(area_style)

    return chart_config


def main():
    domain_results = aggregate_domain_results()
    radar_chart_config = generate_radar_plot(domain_results)
    scatter_plot_data = generate_scatter_plot()
    barplot_data = generate_barplot(domain_results)
    json.dump(
        radar_chart_config,
        open(Path(lambench.__file__).parent / "metrics/results/radar.json", "w"),
        indent=2,
    )
    print("Radar plots saved to radar.json")
    json.dump(
        scatter_plot_data,
        open(Path(lambench.__file__).parent / "metrics/results/scatter.json", "w"),
        indent=2,
    )
    print("Scatter plots saved to scatter.json")
    json.dump(
        barplot_data,
        open(Path(lambench.__file__).parent / "metrics/results/barplot.json", "w"),
        indent=2,
    )
    print("Domain results saved to barplot.json")


if __name__ == "__main__":
    main()
