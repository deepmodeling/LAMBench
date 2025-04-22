import logging
from pathlib import Path
from typing import Optional
import yaml
import lambench
from lambench.databases.calculator_table import CalculatorRecord
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.metrics.post_process import DIRECT_TASK_WEIGHTS
from lambench.metrics.utils import (
    exp_average,
    filter_generalizability_force_field_results,
    get_domain_to_direct_task_mapping,
    get_leaderboard_models,
    aggregated_inference_efficiency_results,
)
from lambench.models.basemodel import BaseLargeAtomModel
import pandas as pd
from datetime import datetime

DOWNSTREAM_TASK_METRICS = yaml.safe_load(
    open(Path(lambench.__file__).parent / "metrics/downstream_tasks_metrics.yml", "r")
)


class ResultsFetcher:
    def __init__(self, timestamp: Optional[datetime] = None):
        self.leaderboard_models = get_leaderboard_models(timestamp=timestamp)

    def aggregate_ood_results_for_one_model(
        self, model: BaseLargeAtomModel
    ) -> dict[str, float]:
        """This function retuns the generalizability test results $\bar{M}_{\text{domain}}$ for a given model across all domains."""
        domain_results = {}
        for domain, tasks in get_domain_to_direct_task_mapping(
            DIRECT_TASK_WEIGHTS
        ).items():
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
                    filter_generalizability_force_field_results(
                        task_result[0].to_dict(), task_config, normalize=True
                    )
                )
            if len(norm_log_results) != len(tasks):
                domain_results[domain] = None
            else:
                domain_result = exp_average(norm_log_results)
                normalized_e = domain_result["energy_rmse_natoms"]
                normalized_f = domain_result["force_rmse"]
                normalized_v = domain_result["virial_rmse_natoms"]
                domain_results[domain] = (
                    0.45 * normalized_e + 0.45 * normalized_f + 0.1 * normalized_v
                    if weight_virial
                    else 0.5 * normalized_e + 0.5 * normalized_f
                )
        return domain_results

    def aggregate_ood_results(self) -> dict[str, dict[str, float]]:
        """This function summarizes the generalizability test results $\bar{M}_{\text{domain}}$ for all models across all domains."""
        results = {}
        for model in self.leaderboard_models:
            results[model.model_metadata.pretty_name] = (
                self.aggregate_ood_results_for_one_model(model)
            )
        return results

    def fetch_stability_results(self) -> dict[str, dict]:
        """This calculates the stability score for a given LAM."""
        stability_results = {}
        for model in self.leaderboard_models:
            task_results = CalculatorRecord.query(
                model_name=model.model_name, task_name="nve_md"
            )
            if len(task_results) != 1:
                logging.warning(
                    f"Expected one record for {model.model_name} and nve_md, but got {len(task_results)}"
                )
                continue
            stability_results[model.model_metadata.pretty_name] = task_results[
                0
            ].metrics

        return stability_results

    def fetch_inference_efficiency_results_for_one_model(
        self, model: BaseLargeAtomModel
    ) -> dict[str, float]:
        """This function returns the inference efficiency results for a given LAM."""
        task_results = CalculatorRecord.query(
            model_name=model.model_name, task_name="inference_efficiency"
        )
        if len(task_results) != 1:
            logging.warning(
                f"Expected one record for {model.model_name} and inference_efficiency, but got {len(task_results)}"
            )
        return aggregated_inference_efficiency_results(task_results[0].metrics)

    def fetch_inference_efficiency_results(self) -> dict[str, dict[str, float]]:
        """This function summarizes the inference efficiency results for all models."""
        results = {}
        for model in self.leaderboard_models:
            results[model.model_metadata.pretty_name] = (
                self.fetch_inference_efficiency_results_for_one_model(model)
            )
        return results

    def fetch_downstream_results(self) -> pd.DataFrame:
        """Returns downstream task results as a DataFrame with models as rows and task metrics as columns."""

        # Initialize an empty DataFrame with model names as index
        model_names = [
            model.model_metadata.pretty_name for model in self.leaderboard_models
        ]
        results_df = pd.DataFrame(index=model_names)

        # Populate the DataFrame
        for task in DOWNSTREAM_TASK_METRICS:
            for model in self.leaderboard_models:
                model_name = model.model_metadata.pretty_name
                task_results = self.fetch_one_downstream_results(task, model)

                if task_results is None:
                    continue

                # Add each metric as a column with task name prefix
                for metric_name, metric_value in task_results.items():
                    column_name = f"{task}::{metric_name}"
                    results_df.at[model_name, column_name] = metric_value
        return results_df

    def fetch_one_downstream_results(
        self, task_name: str, model: BaseLargeAtomModel
    ) -> dict[str, float]:
        """This function returns the downstream task results for a given LAM."""
        task_results = CalculatorRecord.query(
            model_name=model.model_name, task_name=task_name
        )
        if len(task_results) != 1:
            logging.warning(
                f"Expected one record for {model.model_name} and {task_name}, but got {len(task_results)}"
            )
            return None
        return task_results[0].metrics
