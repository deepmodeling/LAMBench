from pathlib import Path

import numpy as np
import yaml

import lambench
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.models.basemodel import BaseLargeAtomModel
from lambench.workflow.entrypoint import MODELS

LEADERBOARD_MODELS = {
    k: v
    for k, v in yaml.safe_load(open(MODELS, "r")).items()
    if any(
        [
            v.get("show_direct_task", False),
            v.get("show_finetune_task", False),
            v.get("show_calculator_task", False),
        ]
    )
}
DIRECT_TASKS = {
    k: v
    for k, v in yaml.safe_load(
        open(Path(lambench.__file__).parent / "metrics/direct_tasks_metrics.yml", "r")
    ).items()
}


class ResultProcessor:
    @staticmethod
    def process_results_for_one_model(model: BaseLargeAtomModel):
        """
        This function fetch and process the raw results from corresponding tables for one model across required tasks.
        """
        single_model_results = {
            "model_name": model.model_name,
            "direct_task_results": {},
            "finetune_task_results": {},
            "calculator_task_results": {},
        }

        if model.show_direct_task:
            direct_task_records = DirectPredictRecord.query(
                model_name=model.model_name
            ).all()
            direct_task_results = {}
            normalized_results = []
            for record in direct_task_records:
                task_name = record.task_name
                task_config = DIRECT_TASKS[task_name]
                task_result, normalized_result = (
                    ResultProcessor.filter_direct_task_results(
                        record.to_dict(), task_config
                    )
                )
                normalized_results.append(normalized_result)
                direct_task_results[task_name] = task_result
            direct_task_results["Weighted"] = ResultProcessor.weighted_average(
                normalized_results, list(task_result.keys())
            )

        if model.show_finetune_task:
            pass
        if model.show_calculator_task:
            raise NotImplementedError("Calculator task is not implemented yet.")

        single_model_results["direct_task_results"] = direct_task_results
        return single_model_results

    @staticmethod
    def filter_direct_task_results(
        task_result: dict, task_config: dict
    ) -> tuple[dict, dict]:
        """
        This function filters the direct task results to keep only the metrics with non-zero weights.
        """
        normalized_result = {}
        metrics = ["energy", "force", "virial"]
        for metric in metrics:
            weight = task_config.get(f"{metric}_weight")
            if weight is None:
                task_result[f"{metric}_rmse_natoms"] = None
                task_result[f"{metric}_mae_natoms"] = None
            else:
                normalized_result[f"{metric}_rmse_natoms"] = (
                    task_result[f"{metric}_rmse_natoms"]
                    / task_config[f"dataset_lstsq_std_{metric}"]
                    * weight
                )
                normalized_result[f"{metric}_mae_natoms"] = (
                    task_result[f"{metric}_mae_natoms"]
                    / task_config[f"dataset_lstsq_std_{metric}"]
                    * weight
                )
        return task_result, normalized_result

    @staticmethod
    def weighted_average(results: list[dict], keys: list[str]) -> dict:
        """
        This function calculates the weighted average of the results.
        """
        weighted_result = {}
        for key in keys:
            weighted_result[key] = np.exp(
                np.mean(np.log([result[key] for result in results if key in result]))
            )
        return weighted_result


if __name__ == "__main__":
    ResultProcessor.process_results_for_one_model()
