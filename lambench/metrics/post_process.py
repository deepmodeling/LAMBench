import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

import lambench
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.databases.calculator_table import CalculatorRecord
from lambench.databases.property_table import PropertyRecord
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.metrics.utils import (
    filter_generalizability_force_field_results,
    exp_average,
    aggregated_nve_md_results,
    aggregated_inference_efficiency_results,
    get_leaderboard_models,
)

DIRECT_TASK_WEIGHTS = yaml.safe_load(
    open(Path(lambench.__file__).parent / "metrics/direct_task_weights.yml", "r")
)
PROPERTY_TASK_MAP = yaml.safe_load(
    open(Path(lambench.__file__).parent / "metrics/finetune_tasks_metrics.yml", "r")
)

PROPERTY_TASK_REVERSE_MAP = {
    v_i: k for k, v in PROPERTY_TASK_MAP.items() for v_i in v["subtasks"]
}


def process_results_for_one_model(model: BaseLargeAtomModel):
    """
    This function fetch and process the raw results from corresponding tables for one model across required tasks.
    """
    single_model_results = {}
    # Generalizability Force Field Task
    if model.show_direct_task:
        single_model_results["generalizability_force_field_results"] = (
            process_force_field_for_one_model(model)
        )

    # Generalizability Domain Specif Task
    if model.show_calculator_task:
        single_model_results["generalizability_domain_specific_results"] = (
            process_domain_specific_for_one_model(model)
        )

    # Adaptability Task
    if model.show_finetune_task:
        single_model_results["adaptability_results"] = (
            process_adaptability_for_one_model(model)
        )

    # Applicability Task
    if model.show_calculator_task:
        single_model_results["applicability_results"] = (
            process_applicability_task_for_one_model(model)
        )

    return single_model_results


def process_force_field_for_one_model(model: BaseLargeAtomModel):
    direct_task_records = DirectPredictRecord.query(model_name=model.model_name)
    if not direct_task_records:
        logging.warning(f"No direct task records found for {model.model_name}")
        return {}

    generalizability_force_field_results = {}
    norm_log_results = []
    for record in direct_task_records:
        if record.task_name not in DIRECT_TASK_WEIGHTS:
            logging.warning(
                f"Deprecated direct task {record.task_name} for {model.model_name}"
            )
            continue
        generalizability_force_field_results[record.task_name] = record.to_dict(
            ev_to_mev=True
        )
        normalized_result = filter_generalizability_force_field_results(
            generalizability_force_field_results[record.task_name],
            DIRECT_TASK_WEIGHTS[record.task_name],
        )
        norm_log_results.append(normalized_result)

    missing_tasks = (
        DIRECT_TASK_WEIGHTS.keys() - generalizability_force_field_results.keys()
    )
    if missing_tasks:
        generalizability_force_field_results["Weighted"] = None
        logging.warning(
            f"Weighted results for {model.model_name} are marked as None due to missing tasks: {missing_tasks}"
        )
    else:
        weighted_results = exp_average(norm_log_results)
        generalizability_force_field_results["Weighted"] = {
            k: np.round(v, 1) if v is not None else None
            for k, v in weighted_results.items()
        }
    return generalizability_force_field_results


def process_domain_specific_for_one_model(model: BaseLargeAtomModel):
    calculator_task_records = CalculatorRecord.query(model_name=model.model_name)
    if not calculator_task_records:
        logging.warning(f"No calculator task records found for {model.model_name}")
        return {}

    applicability_results = {}
    for record in calculator_task_records:
        if record.task_name in [
            "phonon_mdr",
            "torsionnet",
            "neb",
            "wiggle150",
            "elastic",
            "vacancy",
            "binding_energy",
            "rxn_barrier",
            "pressure",
            "stacking_fault",
            "interface",
            "surface",
        ]:
            applicability_results[record.task_name] = record.metrics
    return applicability_results


def process_adaptability_for_one_model(model: BaseLargeAtomModel):
    property_task_records = PropertyRecord.query(model_name=model.model_name)
    if not property_task_records:
        logging.warning(f"No property task records found for {model.model_name}")
        return {}

    property_task_results = defaultdict(list)
    for record in property_task_records:
        property_task_results[PROPERTY_TASK_REVERSE_MAP[record.task_name]].append(
            record.to_dict()
        )

    for task_name, results in property_task_results.items():
        # Missing Data Check
        if len(results) != len(PROPERTY_TASK_MAP[task_name]["subtasks"]):
            logging.warning(f"Missing data for {model.model_name} in {task_name}")
            return {}

        property_task_results[task_name] = {
            metric_name: np.round(
                np.mean([fold_results[metric_name] for fold_results in results]), 7
            )
            for metric_name in results[0]
        }
    # TODO: provide a weighted results for property tasks
    return property_task_results


def process_applicability_task_for_one_model(model: BaseLargeAtomModel):
    calculator_task_records = CalculatorRecord.query(model_name=model.model_name)
    if not calculator_task_records:
        logging.warning(f"No calculator task records found for {model.model_name}")
        return {}

    applicability_results = {}
    for record in calculator_task_records:
        if record.task_name == "nve_md":
            applicability_results[record.task_name] = aggregated_nve_md_results(
                record.metrics
            )
        elif record.task_name == "inference_efficiency":
            applicability_results[record.task_name] = (
                aggregated_inference_efficiency_results(record.metrics)
            )
    return applicability_results


def main():
    results = {}
    leaderboard_models = get_leaderboard_models()
    for model in leaderboard_models:
        r = results[model.model_metadata.pretty_name] = process_results_for_one_model(
            model
        )
        # PosixPath is not JSON serializable
        r["model"] = json.loads(
            json.dumps(model.model_dump(exclude={"model_path"}), default=str)
        )

    with open(
        Path(lambench.__file__).parent / "metrics/results/results.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print("Results saved to results.json")


if __name__ == "__main__":
    main()
