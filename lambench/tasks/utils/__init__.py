import logging
from math import isnan
from pathlib import Path
from typing import Optional


def parse_dptest_log_file(
    filepath: Path, output_type: str = "standard"
) -> dict[str, float | None]:
    """
    Parse dptest results to a dict

    Parameters:
    ----------
        filepath: str
            The path to the dptest output logfile.
        output_type: str
            The type of dptest output file. Options are "standard" and "property".

    """

    # Determine the line after "weighted average of errors" and "number of systems"
    if output_type == "standard":
        content_start_line = -11
    elif output_type == "property":
        content_start_line = -3
    else:
        raise ValueError(f"Unknown dptest output type: {output_type}")

    with open(filepath, "r") as f:
        content = f.readlines()
    metrics = {}
    for line in content[content_start_line:-1]:
        line = line.split("deepmd.entrypoints.test")[-1].strip()
        # Force  MAE/natoms -> force_mae_natoms in compliance with DirectPredictRecord/PropertyRecord
        key = (
            line.split(":")[0]
            .strip()
            .replace("  ", " ")
            .replace(" ", "_")
            .replace("/", "_")
            .lower()
        )
        value = float(line.split(":")[-1].strip().split(" ")[0])
        if isnan(value):
            value = None  # MySQL does not support NaN
        metrics[key] = value
    if all(value is None for value in metrics.values()):
        logging.warning("All metrics are NaN. Something went wrong.")
    return metrics
