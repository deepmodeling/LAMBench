import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from lambench.metrics.vishelper.results_fetcher import DOWNSTREAM_TASK_METRICS
from lambench.metrics.utils import NVEMD_NSTEPS


class MetricsCalculator:
    def __init__(self, fetcher):
        self.fetcher = fetcher

    def calculate_mean_m_bar_domain(self, model) -> float:
        """This calculates $\bar{M}_{\text{domain}}$ for a given LAM across all domains."""
        domain_results = list(
            self.fetcher.aggregate_ood_results_for_one_model(model).values()
        )
        return (
            np.mean(domain_results)
            if None not in domain_results and len(domain_results) > 0
            else None
        )

    def convert_metric_to_score(
        self,
        metric_dict: dict[str, float],
        method: str = "minmax",
    ) -> dict[str, float]:
        """Convert metric values (where lower is better) to normalized scores in range [0, 1] (where higher is better)."""
        if not metric_dict:
            return {}
        scores = {}
        if method == "minmax":
            min_value = min(metric_dict.values())
            max_value = max(metric_dict.values())

            # If all values are the same, return 1.0 for all
            if max_value == min_value:
                return {model: 1.0 for model in metric_dict}
            # Convert: score = 1 - (value - min) / (max - min)
            for model, value in metric_dict.items():
                scores[model] = 1.0 - (value - min_value) / (max_value - min_value)
        elif method == "-log":
            max_value = max([-np.log(value) for value in metric_dict.values()])
            for model, value in metric_dict.items():
                if value == 0:
                    scores[model] = 1
                else:
                    scores[model] = -np.log(value) / max_value
        else:
            raise ValueError(f"Unknown method: {method}")
        return scores

    def calculate_generalizability_ood_error_metric(self) -> dict[str, float]:
        """
        Calculate generalizability error metric on force field prediction tasks
        for each model across all domains.

        Returns a dictionary mapping model names to scores in [0,1] range.
        0 indicates perfect prediction with zero error, while 1 indicates
        performance equivalent to a dummy model.
        """
        # Get model performances by domain
        m_bar_domain = self.fetcher.aggregate_ood_results()
        # filter out models with missing domain results
        m_bar_domain = {
            model: domains
            for model, domains in m_bar_domain.items()
            if None not in domains.values()
        }

        if not m_bar_domain:
            logging.warning("No domain results found.")
            return {}

        # rearrange m_bar_domain {model: {domain: value}} to {model: [value1, value2, ...]}
        rearranged_m_bar_domain = defaultdict(list)
        for model, domains in m_bar_domain.items():
            for domain, value in domains.items():
                rearranged_m_bar_domain[model].append(value)

        return {
            model: np.mean(values) for model, values in rearranged_m_bar_domain.items()
        }

    def calculate_generalizability_downstream_score(self) -> dict[str, float]:
        raw_results = self.fetcher.fetch_downstream_results()

        # Extract necessary columns and prepare penalty dict
        necessary_columns = []
        penalty_dict = {}
        domain_columns = defaultdict(list)  # use in domain level aggregation
        for task_name, task_config in DOWNSTREAM_TASK_METRICS.items():
            # Add all metric columns
            metrics_columns = [
                f"{task_name}::{metrics_name}"
                for metrics_name in task_config["metrics"]
            ]
            domain_columns[task_config["domain"]].extend(metrics_columns)
            necessary_columns.extend(metrics_columns)

            # Add penalty column and mapping if available
            if "penalty" in task_config:
                penalty_column = f"{task_name}::{task_config['penalty']}"
                necessary_columns.append(penalty_column)
                penalty_dict[penalty_column] = metrics_columns

        # Filter dataframe to include only the necessary columns
        raw_results = raw_results[necessary_columns]

        # Normalize all metrics by dummy baseline dimensionless metrics
        # This gives values equivalent to $\bar{M}_i$, where $i$ is the task name, no log average needed
        for column in raw_results.columns:
            if column not in penalty_dict:
                check_dummy = column.split(
                    "::"
                )  # split the column name back to task name and metric name
                assert len(check_dummy) == 2, (
                    f"Column name {column} is not in the expected format"
                )
                if (
                    DOWNSTREAM_TASK_METRICS[check_dummy[0]].get("dummy", None)
                    is not None
                ):
                    dummy = DOWNSTREAM_TASK_METRICS[check_dummy[0]]["dummy"][
                        check_dummy[1]
                    ]
                else:
                    dummy = None

                if dummy is None:
                    logging.warning(
                        f"Dummy value not found for {column}, skipping normalization"
                    )
                else:
                    # normalize the metric by dummy value, TODO: check if dummy is 0
                    raw_results[column] = raw_results[column] / dummy

        # Apply penalty for specified metrics directly to $\bar{M}_i$ before domain level aggregation.
        # $\bar{M}_i$ is an error metric, the lower the better, so we want to penalize it by dividing
        # by the penalty column (success rate in range [0,1])
        for penalty_column, metrics_to_penalize in penalty_dict.items():
            for metric in metrics_to_penalize:
                if metric not in raw_results.columns:
                    logging.error(f"Metric {metric} not found in raw results")
                    continue
                raw_results[metric] = raw_results[metric] / raw_results[penalty_column]
        # Aggregate all metrics for each domain to get domain level error metrics equivalent to $\bar{M}_{\text{domain}}$
        domain_level_metrics = {}
        for domain, columns in domain_columns.items():
            domain_df = raw_results[columns]
            domain_level_metrics[domain] = domain_df.mean(axis=1)
        domain_results = pd.DataFrame(domain_level_metrics)

        # # Now aggregate all domains to get the final generalizability metrics for each model
        return domain_results.mean(axis=1).to_dict()

    def calculate_stability_results(self) -> dict[str, float]:
        """This calculates the stability score for a given LAM."""
        stability_results = self.fetcher.fetch_stability_results()
        # filter out models with missing stability results
        stability_results = {
            model: metrics for model, metrics in stability_results.items()
        }

        stability_results = pd.DataFrame.from_dict(stability_results, orient="index")
        stability_results = stability_results.applymap(
            lambda cell: self._calculate_instability_error(cell)
        )
        # average over all systems
        stability_results = stability_results.mean(axis=1)
        return stability_results.to_dict()

    def _calculate_instability_error(self, cell: dict, lambda_0: float = 5e-4) -> float:
        """
        Private method applied to each cell of the stability_results DataFrame.
        Calculates the instability error for a given LAM on a given NVE traj.
        Penalty is applied if the simulation is not complete, by returning a large value.
        """

        if cell is None or cell.get("steps", None) is None:
            return 5
        if cell["steps"] != NVEMD_NSTEPS:
            return 5
        else:
            slope = cell.get("slope", None)
            if slope is None:
                return 5
            else:
                return np.clip(np.log10(cell["slope"] / lambda_0), a_min=0, a_max=None)

    def calculate_efficiency_results(self) -> dict[str, float]:
        efficiency_results = self.fetcher.fetch_inference_efficiency_results()
        # filter out models with missing efficiency results
        efficiency_results = {
            model: metrics
            for model, metrics in efficiency_results.items()
            if metrics is not None and metrics["average_time"] is not None
        }
        if not efficiency_results:
            logging.warning("No inference efficiency results found.")
            return {}

        # extract inference time and calculate efficiency score
        efficiency_results = {
            model: 100 / metrics["average_time"]
            for model, metrics in efficiency_results.items()
        }
        return efficiency_results

    def summarize_final_rankings(self):
        generalizability_ood = self.calculate_generalizability_ood_error_metric()
        generalizability_downstream = self.calculate_generalizability_downstream_score()
        stability_results = self.calculate_stability_results()
        efficiency_results = self.calculate_efficiency_results()
        if not generalizability_ood or not generalizability_downstream:
            logging.warning(
                "Missing data for generalizability metrics (ood or downstream)"
            )
            return

        shared_models = (
            set(generalizability_ood.keys())
            .intersection(set(generalizability_downstream.keys()))
            .intersection(set(stability_results.keys()))
            .intersection(set(efficiency_results.keys()))
        )
        if not shared_models:
            logging.warning(
                "No models have both generalizability and applicability metrics"
            )
            return

        # Create multi-level column DataFrame
        data = {
            "Generalizability-FF Error ↓": [
                generalizability_ood[model] for model in shared_models
            ],
            "Generalizability-DS Error ↓": [
                generalizability_downstream[model] for model in shared_models
            ],
            "Applicability-Instability ↓": [
                stability_results[model] for model in shared_models
            ],
            "Applicability-Efficiency ↑": [
                efficiency_results[model] for model in shared_models
            ],
        }

        # Create DataFrame with models as index
        summary_df = pd.DataFrame(data, index=list(shared_models))
        summary_df.index.name = "Model"

        # Reset index to make 'Model' a regular column
        summary_df = summary_df.reset_index()

        summary_df.reset_index(drop=True, inplace=True)

        summary_df = summary_df.round(3)
        summary_df = summary_df.sort_values(
            by=[
                "Generalizability-FF Error ↓",
                "Generalizability-DS Error ↓",
                "Applicability-Instability ↓",
                "Applicability-Efficiency ↑",
            ],
            ascending=[True, True, True, False],
        )
        print(
            "Final Rankings:\n",
            summary_df.to_string(index=False),
        )
        return summary_df
