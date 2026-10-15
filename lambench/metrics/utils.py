import json
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

import lambench
from lambench.workflow.entrypoint import gather_model, gather_model_params

_DIATOMICS_JSON = (
    Path(__file__).resolve().parent.parent
    / "tasks"
    / "calculator"
    / "diatomics"
    / "diatomics.json"
)

#############################
# General utility functions #
#############################


def get_leaderboard_models(timestamp: datetime | None = None) -> list:
    models = [gather_model(param, "") for param in gather_model_params()]
    if timestamp is not None:
        models = [
            model for model in models if model.model_metadata.date_added <= timestamp
        ]
    return [
        model
        for model in models
        if model.show_direct_task
        or model.show_finetune_task
        or model.show_calculator_task
    ]


def exp_average(log_results: list[dict]) -> dict[str, float | None]:
    """Calculate the exponential average of each metric of the results."""
    exp_average_metrics = {}
    all_keys = set([key for result in log_results for key in result.keys()])
    for key in sorted(all_keys):
        try:
            metrics_list = [result[key] for result in log_results]
        except KeyError:
            # Contains None(NaN) for metrics with weight != None;
            # For the comparability among tasks, set it to None
            exp_average_metrics[key] = None
            continue
        # Filter out "legal" None values with weight == None
        metrics_list = [m for m in metrics_list if m is not None]
        if len(metrics_list) == 0:
            exp_average_metrics[key] = None
            continue
        exp_average_metrics[key] = np.round(np.exp(np.mean(metrics_list)), 7)
    return exp_average_metrics


#################################
# Direct Task utility functions #
#################################


def filter_generalizability_force_field_results(
    task_result: dict, task_config: dict, normalize: bool | None = False
) -> dict:
    """
    This function filters the direct task results to keep only the metrics with non-zero task weights.

    I. Optional: normalize the metrics by multiply {metric}_std. (Required for Property)
    II. Remove tasks where weight is None in the DIRECT_TASK_METRICS.
        Please note that this change also applies in the input dict.
    III. Calculate the weighted **log** metrics.

    NOTE: We normalize first to ensure the weight is a dimensionless number.

    Returns: metrics for each task normalized, logged, and weighted.
    """
    filtered_metrics = {}
    for k, v in task_result.items():
        efvp: Literal["energy", "force", "virial", "property"] = k.split("_")[0]
        weight = task_config.get(f"{efvp}_weight")
        if weight is None:
            filtered_metrics[k] = None
            task_result[k] = None
            continue
        std = task_config.get(f"{efvp}_std")

        if v is not None:
            if normalize:
                v = np.min(
                    [v / std, 1]
                )  # cap the normalized value to 1, for models worese than a dummy baseline, use dummy baseline.
            filtered_metrics[k] = np.log(v) * weight
            # else the filtered_metrics will not have this key.
            # Metrics with weight != None should have a value,
            # Or the weighted result would be marked to None.
    return filtered_metrics


#####################################
# Calculator Task utility functions #
#####################################

## NVE MD utility functions
CALCULATOR_TASKS = yaml.safe_load(
    open(Path(lambench.__file__).parent / "tasks/calculator/calculator_tasks.yml", "r")
)
NVEMD_NSTEPS = CALCULATOR_TASKS["nve_md"]["calculator_params"]["num_steps"]


def aggregated_nve_md_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    This function aggregates the NVE MD results from multiple systems for one LAM.
    It calculates the average and standard deviation of each metric across systems,
    and returns the aggregated results.
    """
    aggregated_result = {}
    success_count = len(results)
    for test_system, result in results.items():
        if result["steps"] != NVEMD_NSTEPS or result["slope"] >= 50:
            success_count -= 1
            continue  # Skip the incomplete simulation
        for k, v in result.items():
            if k not in aggregated_result:
                aggregated_result[k] = []
            if v is None:
                v = np.nan
            aggregated_result[k].append(v)
    for k, v in aggregated_result.items():
        aggregated_result[k] = np.round(np.exp(np.mean(np.log(v))), 6)
    aggregated_result["success_rate"] = np.round(success_count / len(results), 2)
    return aggregated_result


## Inference efficiency utility functions
def aggregated_inference_efficiency_results(
    results: dict[str, dict[str, float]],
) -> dict[str, float]:
    system_level_avg = []
    system_level_std = []
    system_level_success_rate = []
    success_count = len(results)
    for _, result in results.items():
        if result["average_time"] is None:
            success_count -= 1
            continue
        system_level_avg.append(result["average_time"])
        system_level_std.append(result["std_time"])
        system_level_success_rate.append(result["success_rate"])
    if success_count != len(results):
        return {"average_time": None, "std_time": None, "success_rate": 0.0}
    return {
        "average_time": np.round(np.mean(system_level_avg), 6),
        "standard_deviation": np.round(
            np.sqrt(np.mean(np.square(system_level_std))), 6
        ),
        "success_rate": np.round(np.mean(system_level_success_rate), 2),
    }


@lru_cache
def _diatomics_molecule_names() -> tuple[str, ...]:
    with open(_DIATOMICS_JSON) as fh:
        return tuple(entry["name"] for entry in json.load(fh))


def _empty_diatomics_agg() -> dict[str, float | None]:
    return {
        "avg_roughness": None,
        "bond_length_mae": None,
        "well_depth_mae": None,
        "bond_length_score": None,
        "well_depth_score": None,
        "score": None,
    }


def _diatomic_dummy_score(error: float, dummy_error: float) -> float:
    # A zero-error dummy is already exact: only an exact prediction matches it.
    if dummy_error == 0:
        return 0.0 if error == 0 else 1.0
    return float(min(error / dummy_error, 1.0))


def aggregated_diatomics_results(results: dict) -> dict[str, float | None]:
    """
    Aggregate per-molecule diatomics results.

    avg_roughness: mean slope MAE relative to a constant dummy, capped at 1.
    New results carry the tested molecule set and per-molecule dummy errors.
    The three scores are averaged with equal weight only for complete results.
    Legacy roughness-only records remain readable but have no combined score.
    """
    if not results:
        return _empty_diatomics_agg()

    names = results.get("reference_molecules", _diatomics_molecule_names())
    curves = results.get("curves", results)
    if not names:
        return _empty_diatomics_agg()

    roughness_values = []
    errors = {"bond_length": [], "well_depth": []}
    scores = {"bond_length": [], "well_depth": []}
    complete_metrics = True
    for name in names:
        mol_results = curves.get(name)
        if mol_results is None:
            return _empty_diatomics_agg()
        roughness = mol_results.get("roughness")
        if roughness is None or not np.isfinite(roughness) or not 0 <= roughness <= 1:
            return _empty_diatomics_agg()
        roughness_values.append(roughness)
        for metric in errors:
            error = mol_results.get(f"{metric}_error")
            dummy = mol_results.get(f"{metric}_dummy_error")
            if metric == "bond_length" and error is None and dummy is None:
                # Only references below the binding threshold skip bond position.
                depth = mol_results.get("well_depth_dummy_error")
                if depth is None or depth >= 0.05:
                    complete_metrics = False
                continue
            if (
                error is None
                or dummy is None
                or not np.isfinite(error)
                or not np.isfinite(dummy)
                or error < 0
                or dummy < 0
            ):
                complete_metrics = False
                continue
            errors[metric].append(error)
            scores[metric].append(_diatomic_dummy_score(error, dummy))

    aggregated = _empty_diatomics_agg()
    aggregated["avg_roughness"] = float(np.mean(roughness_values))
    for metric in errors:
        if errors[metric]:
            aggregated[f"{metric}_mae"] = float(np.mean(errors[metric]))
            aggregated[f"{metric}_score"] = float(np.mean(scores[metric]))
    terms = [
        aggregated["avg_roughness"],
        aggregated["bond_length_score"],
        aggregated["well_depth_score"],
    ]
    if complete_metrics and all(term is not None for term in terms):
        aggregated["score"] = float(np.mean(terms))
    return aggregated


####################################
# Visualization utility functions #
####################################

## Radar plot utility functions


def get_domain_to_direct_task_mapping(config_file: dict) -> dict:
    """
    This function fetches the domain to direct task mapping from the config file.
    """
    domain_to_direct_task_mapping = defaultdict(list)
    for task, task_config in config_file.items():
        domain = task_config["domain"]
        domain_to_direct_task_mapping[domain].append(task)
    return domain_to_direct_task_mapping
