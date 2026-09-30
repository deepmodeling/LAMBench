from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

import lambench
from lambench.workflow.entrypoint import gather_model, gather_model_params

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
) -> dict[str, float | None]:
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
        "average_time": float(np.round(np.mean(system_level_avg), 6)),
        "standard_deviation": float(
            np.round(np.sqrt(np.mean(np.square(system_level_std))), 6)
        ),
        "success_rate": float(np.round(np.mean(system_level_success_rate), 2)),
    }


## Diatomic utility functions
def _empty_diatomics_agg() -> dict[str, float | None]:
    return {
        "bond_length_mae": None,
        "well_depth_mae": None,
        "force_flip_count": None,
        "coverage": 0.0,
        "score": None,
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _dummy_hat(value: float | None, dummy: float) -> float | None:
    if value is None or not np.isfinite(value) or not np.isfinite(dummy) or dummy <= 0:
        return None
    return float(min(value / dummy, 1.0))


def _element_is_excluded(record: dict | None) -> bool:
    if not isinstance(record, dict) or record.get("model_excluded", True):
        return True
    flip = record.get("force_flip_count")
    return flip is None or not np.isfinite(flip)


def aggregated_diatomics_results(results: dict[str, dict]) -> dict[str, float | None]:
    """Aggregate per-element diatomic curves into one coverage-weighted score.

    bond_length_mae is in Å and well_depth_mae is in eV. force_flip_count is
    the mean number of sign changes in first-atom Fx. Bond and well errors are
    divided by their blind dummy scales; force flips use the mean per-element
    absolute deviation from the ideal count of one. Each term is capped at 1.
    score is their equal-weight average divided by coverage. Missing elements
    and model_excluded curves lower coverage instead of discarding the model.
    score is None when coverage is zero or any term has no finite samples.
    """
    from lambench.tasks.calculator.diatomics.diatomics import (
        reference_dummy_scales,
        scored_element_names,
    )

    names = scored_element_names()
    if not names:
        return _empty_diatomics_agg()

    bond_errors: list[float] = []
    well_errors: list[float] = []
    flip_counts: list[float] = []
    flip_deviations: list[float] = []
    n_excluded = 0
    for name in names:
        record = None if not results else results.get(name)
        if _element_is_excluded(record):
            n_excluded += 1
            continue
        assert isinstance(record, dict)
        bond = record.get("bond_length_error")
        well = record.get("well_depth_error")
        if bond is not None and np.isfinite(bond):
            bond_errors.append(float(bond))
        if well is not None and np.isfinite(well):
            well_errors.append(float(well))
        flip_count = float(record["force_flip_count"])
        flip_counts.append(flip_count)
        flip_deviations.append(abs(flip_count - 1.0))

    coverage = (len(names) - n_excluded) / len(names)
    bond_mae = _mean_or_none(bond_errors)
    well_mae = _mean_or_none(well_errors)
    flip_count = _mean_or_none(flip_counts)
    flip_deviation = _mean_or_none(flip_deviations)
    dummy = reference_dummy_scales()
    bond_hat = _dummy_hat(bond_mae, dummy["bond_length_mae"])
    well_hat = _dummy_hat(well_mae, dummy["well_depth_mae"])
    flip_hat = _dummy_hat(flip_deviation, dummy["force_flip_deviation"])
    if coverage <= 0 or bond_hat is None or well_hat is None or flip_hat is None:
        score = None
    else:
        score = float((bond_hat + well_hat + flip_hat) / (3.0 * coverage))
    return {
        "bond_length_mae": bond_mae,
        "well_depth_mae": well_mae,
        "force_flip_count": flip_count,
        "coverage": float(coverage),
        "score": score,
    }


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
