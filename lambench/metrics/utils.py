import numpy as np
import yaml
from typing import Optional, Literal
import lambench
from pathlib import Path


# General utility functions
def exp_average(log_results: list[dict]) -> dict[str, Optional[float]]:
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
        exp_average_metrics[key] = np.round(np.exp(np.mean(metrics_list)), 7)
    return exp_average_metrics


# Direct Task utility functions
def filter_direct_task_results(
    task_result: dict, task_config: dict, normalize: Optional[bool] = False
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
        if normalize and std is not None:
            weight /= std

        if v is not None:
            filtered_metrics[k] = np.log(v) * weight
            # else the filtered_metrics will not have this key.
            # Metrics with weight != None should have a value,
            # Or the weighted result would be marked to None.
    return filtered_metrics


# Calculator Task utility functions
## NVE MD utility functions
NVEMD_NSTEPS = yaml.safe_load(
    open(Path(lambench.__file__).parent / "tasks/calculator/calculator_tasks.yml", "r")
)["nve_md"]["calculator_params"]["num_steps"]


def aggregated_nve_md_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
    aggregated_result = {}
    success_count = len(results)
    for test_system, result in results.items():
        if result["steps"] != NVEMD_NSTEPS:
            success_count -= 1
            continue  # Skip the incomplete simulation
        for k, v in result.items():
            if k not in aggregated_result:
                aggregated_result[k] = []
            aggregated_result[k].append(v)
    for k, v in aggregated_result.items():
        aggregated_result[k] = np.round(np.exp(np.mean(np.log(v))), 6)
    aggregated_result["success_rate"] = np.round(success_count / len(results), 2)
    return aggregated_result


# Metrics metadata
METRICS_METADATA = (
    {
        "direct_task_results": {
            "DISPLAY_NAME": "Direct Prediction Accuracy",
            "DESCRIPTION": "Energy, force, and virial prediction accuracy of the model on the test sets accross multiple domains.",
            "HPt_NC_2022": {
                "DISPLAY_NAME": "H/Pt Catalysis",
                "DESCRIPTION": "A direct two-phase simulation of heterogeneous turnover on the catalyst surface at chemical accuracy. [https://arxiv.org/abs/2106.01949]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "MD22": {
                "DISPLAY_NAME": "MD22",
                "DESCRIPTION": "Dataset containing MD trajectories of the 42-atom tetrapeptide Ac-Ala3-NHMe from the MD22 benchmark set. [https://www.science.org/doi/10.1126/sciadv.adf0873]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "Ca_batteries_CM2021": {
                "DISPLAY_NAME": "Ca Batteries",
                "DESCRIPTION": "A dataset of Ca-bearing minerals. [https://www.nature.com/articles/s41598-019-46002-4]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "NequIP_NC_2022": {
                "DISPLAY_NAME": "NequIP2022",
                "DESCRIPTION": "Approximately 57,000 configurations from the evaluation datasets for NequIP graph neural network model for interatomic potentials. Trajectories have been taken from LiPS, LiPO glass melt-quench simulation, and formate decomposition on Cu datasets. [https://www.nature.com/articles/s41467-022-29939-5]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "ANI": {
                "DISPLAY_NAME": "ANI",
                "DESCRIPTION": "The training dataset of ANI-1x model. 5.5 M structures. [https://doi.org/10.1063/1.5023802]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "REANN_CO2_Ni100": {
                "DISPLAY_NAME": "REANN",
                "DESCRIPTION": "Example training data of the REANN package. [https://doi.org/10.1103/PhysRevLett.127.156002]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "Collision": {
                "DISPLAY_NAME": "Collision",
                "DESCRIPTION": "Validation set from COLL. Consists of configurations taken from molecular collisions of different small organic molecules. [https://arxiv.org/abs/2011.14115]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "CGM_MLP_NC2023": {
                "DISPLAY_NAME": "Carbon Deposition",
                "DESCRIPTION": "Dynamic simulations of carbon deposition on metal surfaces like Cu(111), Cr(110), Ti(001), and oxygen-contaminated Cu(111). [https://www.nature.com/articles/s41467-023-44525-z]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "Subalex_9k": {
                "DISPLAY_NAME": "SubAlex_9k",
                "DESCRIPTION": "A dataset of 9k structures from the SubAlex dataset. [https://arxiv.org/abs/2410.12771]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "Torsionnet500": {
                "DISPLAY_NAME": "Torsionnet500",
                "DESCRIPTION": "TorsionNet500, a benchmark data set comprising 500 chemically diverse fragments with DFT torsion profiles (12k MM- and DFT-optimized geometries and energies). [https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c01346]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "Cu_MgO_catalysts": {
                "DISPLAY_NAME": "Cu-MgO-Al2O3 Catalysts",
                "DESCRIPTION": "Selective CO2 Hydrogenation to Methanol over Cu-MgO-Al2O3 Catalysts. [https://pubs.acs.org/doi/10.1021/jacs.3c10685]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "Si_ZEO22": {
                "DISPLAY_NAME": "Si-ZEO22",
                "DESCRIPTION": "Dataset consisting of 350000 DFT single point energy calculations from 219 different pure silica zeolite topologies. [https://github.com/tysours/Si-ZEO22]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "AIMD-Chig": {
                "DISPLAY_NAME": "AIMD-Chig",
                "DESCRIPTION": "MD dataset including 2 million conformations of 166-atom protein Chignolin sampled at the density functional theory (DFT) level. [https://www.nature.com/articles/s41597-023-02465-9]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "HEA25_S": {
                "DISPLAY_NAME": "High Entropy Alloys Surfaces",
                "DESCRIPTION": "A dataset of 25-atom high entropy alloy surfaces, focusing on 25 d-block transition metals, excluding Tc, Cd, Re, Os and Hg. [https://arxiv.org/abs/2212.13254]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "HEMC_HEMB": {
                "DISPLAY_NAME": "HEMC_HEMB",
                "DESCRIPTION": "DFT dataset of high-entropy transition metal diboride (HEMB2) ceramics and high-entropy transition metal carbide (HEMC) ceramics. [https://www.oaepublish.com/articles/jmi.2024.14]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "HEA25_bulk": {
                "DISPLAY_NAME": "High Entropy Alloys Bulk",
                "DESCRIPTION": "A dataset of 25-atom high entropy alloy bulk structures, focusing on 25 d-block transition metals, excluding Tc, Cd, Re, Os and Hg. [https://arxiv.org/abs/2212.13254]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "WBM_downsampled": {
                "DISPLAY_NAME": "WBM_downsampled",
                "DESCRIPTION": "A dataset of 25,696 structures from the WBM dataset. [https://www.nature.com/articles/s41524-020-00481-6]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "H_nature_2022": {
                "DISPLAY_NAME": "Hydrogen Combusiton",
                "DESCRIPTION": "Dataset of hydrogen combustion reactions. [https://www.nature.com/articles/s41597-022-01330-5]",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
            "Weighted": {
                "DISPLAY_NAME": "Weighted Results",
                "DESCRIPTION": "Weighted results of all the aformentioned datasets.",
                "energy_rmse": {
                    "DISPLAY_NAME": "Energy RMSE",
                    "DESCRIPTION": "The root mean squared error of the energy prediction, in eV.",
                },
                "energy_mae": {
                    "DISPLAY_NAME": "Energy MAE",
                    "DESCRIPTION": "The mean absolute error of the energy prediction, in eV.",
                },
                "energy_rmse_natoms": {
                    "DISPLAY_NAME": "Energy RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the energy prediction per atom, in eV/atom.",
                },
                "energy_mae_natoms": {
                    "DISPLAY_NAME": "Energy MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the energy prediction per atom, in eV/atom.",
                },
                "force_rmse": {
                    "DISPLAY_NAME": "Force RMSE",
                    "DESCRIPTION": "The root mean squared error of the force prediction, in eV/Å.",
                },
                "force_mae": {
                    "DISPLAY_NAME": "Force MAE",
                    "DESCRIPTION": "The mean absolute error of the force prediction, in eV/Å.",
                },
                "virial_rmse": {
                    "DISPLAY_NAME": "Virial RMSE",
                    "DESCRIPTION": "The root mean squared error of the virial prediction, in eV.",
                },
                "virial_mae": {
                    "DISPLAY_NAME": "Virial MAE",
                    "DESCRIPTION": "The mean absolute error of the virial prediction, in eV.",
                },
                "virial_rmse_natoms": {
                    "DISPLAY_NAME": "Virial RMSE per Atom",
                    "DESCRIPTION": "The root mean squared error of the virial prediction per atom, in eV/atom.",
                },
                "virial_mae_natoms": {
                    "DISPLAY_NAME": "Virial MAE per Atom",
                    "DESCRIPTION": "The mean absolute error of the virial prediction per atom, in eV/atom.",
                },
            },
        },
        "finetune_task_results": {
            "DISPLAY_NAME": "Property Finetune Accuracy",
            "DESCRIPTION": "Accuracy of the property finetuning task. Note: The test results does not represent the converged accuracy due to limited fine-tuning steps.",
            "Matbench_mp_e_form": {
                "DISPLAY_NAME": "Matbench_mp_e_form",
                "DESCRIPTION": "Formation energy of materials, in eV/atom. 5-fold average.",
            },
            "Matbench_mp_gap": {
                "DISPLAY_NAME": "Matbench_mp_gap",
                "DESCRIPTION": "Band gap of materials, in eV. 5-fold average.",
            },
            "Matbench_jdft2d": {
                "DISPLAY_NAME": "Matbench_jdft2d",
                "DESCRIPTION": "2D exfoliation energy from JARVIS-DFT dataset, in meV/atom. 5-fold average.",
            },
            "Matbench_phonons": {
                "DISPLAY_NAME": "Matbench_phonons",
                "DESCRIPTION": "Frequency of the highest frequency optical phonon mode peak, in units of 1/cm. 5-fold average.",
            },
            "Matbench_dielectric": {
                "DISPLAY_NAME": "Matbench_dielectric",
                "DESCRIPTION": "Refractive index of materials, unitless. 5-fold average.",
            },
            "Matbench_log_kvrh": {
                "DISPLAY_NAME": "Matbench_log_kvrh",
                "DESCRIPTION": "Logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa. 5-fold average.",
            },
            "Matbench_log_gvrh": {
                "DISPLAY_NAME": "Matbench_log_gvrh",
                "DESCRIPTION": "Logarithm of the DFT Voigt-Reuss-Hill average shear moduli in GPa. 5-fold average.",
            },
            "Matbench_perovskites": {
                "DISPLAY_NAME": "Matbench_perovskites",
                "DESCRIPTION": "Formation energy of perovskites, in eV/unit cell. 5-fold average.",
            },
        },
        "calculator_task_results": {
            "DISPLAY_NAME": "Miscellaneous",
            "DESCRIPTION": "Evaluation metrics for miscellaneous tasks.",
            "nve_md": {
                "DISPLAY_NAME": "NVE MD",
                "DESCRIPTION": "NVE Molecular Dynamics simulation.",
                "slope": {
                    "DISPLAY_NAME": "Energy Drift Slope",
                    "DESCRIPTION": "The slope of the energy drift, in eV/atom/ps.",
                },
                "steps": {
                    "DISPLAY_NAME": "Simulation Steps",
                    "DESCRIPTION": "The number of simulation steps.",
                },
                "momenta_diff": {
                    "DISPLAY_NAME": "Momentum Difference",
                    "DESCRIPTION": "The mean momentum difference, in amu · Å/fs.",
                },
                "simulation_time": {
                    "DISPLAY_NAME": "Simulation Efficiency",
                    "DESCRIPTION": "The time taken to simulate 10 ps, in seconds.",
                },
                "success_rate": {
                    "DISPLAY_NAME": "Success Rate",
                    "DESCRIPTION": "The success rate of the simulation over 9 test cases.",
                },
            },
        },
    },
)
