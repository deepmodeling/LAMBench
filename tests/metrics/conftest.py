import pytest
from lambench.databases.direct_predict_table import DirectPredictRecord
from lambench.databases.calculator_table import CalculatorRecord
from lambench.metrics.vishelper.results_fetcher import DOWNSTREAM_TASK_METRICS
from unittest.mock import patch

RECORDS_DIRECT = [
    DirectPredictRecord(
        id=1,
        model_name="test_dp",
        task_name="ANI",
        create_time=None,
        energy_rmse=0.467693,
        energy_mae=0.340148,
        energy_rmse_natoms=0.0376691,
        energy_mae_natoms=0.0227907,
        force_rmse=0.437947,
        force_mae=0.253533,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=2,
        model_name="test_dp",
        task_name="HEA25_S",
        create_time=None,
        energy_rmse=8.0143,
        energy_mae=6.62737,
        energy_rmse_natoms=0.191712,
        energy_mae_natoms=0.159862,
        force_rmse=0.30147,
        force_mae=0.164684,
        virial_rmse=186.408,
        virial_mae=84.1395,
        virial_rmse_natoms=4.42881,
        virial_mae_natoms=2.02307,
    ),
    DirectPredictRecord(
        id=3,
        model_name="test_dp",
        task_name="HEA25_bulk",
        create_time=None,
        energy_rmse=5.76976,
        energy_mae=3.89511,
        energy_rmse_natoms=0.134269,
        energy_mae_natoms=0.0909164,
        force_rmse=0.345338,
        force_mae=0.209072,
        virial_rmse=167.769,
        virial_mae=63.4004,
        virial_rmse_natoms=3.87829,
        virial_mae_natoms=1.47443,
    ),
    DirectPredictRecord(
        id=4,
        model_name="test_dp",
        task_name="HEMC_HEMB",
        create_time=None,
        energy_rmse=7.87692,
        energy_mae=4.38965,
        energy_rmse_natoms=0.154871,
        energy_mae_natoms=0.0900861,
        force_rmse=0.19703,
        force_mae=0.121378,
        virial_rmse=6.12989,
        virial_mae=2.92621,
        virial_rmse_natoms=0.127266,
        virial_mae_natoms=0.0619396,
    ),
    DirectPredictRecord(
        id=5,
        model_name="test_dp",
        task_name="MD22",
        create_time=None,
        energy_rmse=0.13579,
        energy_mae=0.10127,
        energy_rmse_natoms=0.00221771,
        energy_mae_natoms=0.00175116,
        force_rmse=0.0924748,
        force_mae=0.0542347,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=8,
        model_name="test_dp",
        task_name="REANN_CO2_Ni100",
        create_time=None,
        energy_rmse=17.596,
        energy_mae=14.1638,
        energy_rmse_natoms=0.451179,
        energy_mae_natoms=0.363173,
        force_rmse=0.221132,
        force_mae=0.133015,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=9,
        model_name="test_dp",
        task_name="NequIP_NC_2022",
        create_time=None,
        energy_rmse=0.154749,
        energy_mae=0.102555,
        energy_rmse_natoms=0.000866307,
        energy_mae_natoms=0.000654184,
        force_rmse=0.112146,
        force_mae=0.0743181,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=10,
        model_name="test_dp",
        task_name="AIMD-Chig",
        create_time=None,
        energy_rmse=0.734055,
        energy_mae=0.572073,
        energy_rmse_natoms=0.00442202,
        energy_mae_natoms=0.00344622,
        force_rmse=0.266102,
        force_mae=0.167555,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=12,
        model_name="test_dp",
        task_name="Cu_MgO_catalysts",
        create_time=None,
        energy_rmse=0.267982,
        energy_mae=0.153377,
        energy_rmse_natoms=0.0035446,
        energy_mae_natoms=0.00229624,
        force_rmse=0.0584197,
        force_mae=0.038047,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=13,
        model_name="test_dp",
        task_name="Ca_batteries_CM2021",
        create_time=None,
        energy_rmse=1.43961,
        energy_mae=0.614781,
        energy_rmse_natoms=0.0181229,
        energy_mae_natoms=0.0072411,
        force_rmse=0.13488,
        force_mae=0.0946354,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=14,
        model_name="test_dp",
        task_name="HPt_NC_2022",
        create_time=None,
        energy_rmse=0.205067,
        energy_mae=0.169921,
        energy_rmse_natoms=0.00282596,
        energy_mae_natoms=0.00236338,
        force_rmse=0.0626074,
        force_mae=0.0461952,
        virial_rmse=43.0755,
        virial_mae=23.5309,
        virial_rmse_natoms=0.589964,
        virial_mae_natoms=0.328104,
    ),
    DirectPredictRecord(
        id=15,
        model_name="test_dp",
        task_name="Si_ZEO22",
        create_time=None,
        energy_rmse=1.2949,
        energy_mae=0.74051,
        energy_rmse_natoms=0.00893499,
        energy_mae_natoms=0.00544244,
        force_rmse=0.182507,
        force_mae=0.0328574,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=16,
        model_name="test_dp",
        task_name="WBM_downsampled",
        create_time=None,
        energy_rmse=0.194829,
        energy_mae=0.0604359,
        energy_rmse_natoms=0.0318466,
        energy_mae_natoms=0.00875549,
        force_rmse=None,
        force_mae=None,
        virial_rmse=None,
        virial_mae=None,
        virial_rmse_natoms=None,
        virial_mae_natoms=None,
    ),
    DirectPredictRecord(
        id=17,
        model_name="test_dp",
        task_name="Subalex_9k",
        create_time=None,
        energy_rmse=1.90841,
        energy_mae=0.268596,
        energy_rmse_natoms=0.234027,
        energy_mae_natoms=0.0286509,
        force_rmse=0.624174,
        force_mae=0.0437039,
        virial_rmse=4.16581,
        virial_mae=0.373998,
        virial_rmse_natoms=0.382751,
        virial_mae_natoms=0.0371473,
    ),
]

RECORDS_CALCULATOR = [
    CalculatorRecord(
        id=idx,
        model_name="test_dp",
        task_name=task,
        create_time=None,
        metrics=DOWNSTREAM_TASK_METRICS[task]["dummy"],
    )
    for idx, task in enumerate(DOWNSTREAM_TASK_METRICS)
]


def create_query_side_effect(records):
    """
    Create a query side effect function based on the given records

    Args:
        records: List of record objects that have model_name and task_name attributes

    Returns:
        Function that filters records based on model_name and task_name
    """

    def query_side_effect(*args, **kwargs):
        model_name = kwargs.get("model_name")
        task_name = kwargs.get("task_name")

        if task_name is None:
            return [rec for rec in records if rec.model_name == model_name]
        return [
            rec
            for rec in records
            if rec.model_name == model_name and rec.task_name == task_name
        ]

    return query_side_effect


# For backward compatibility, provide specific fixtures
@pytest.fixture
def mock_direct_predict_query():
    """Fixture to parameterize DirectPredictRecord.query calls."""
    with patch("lambench.metrics.post_process.DirectPredictRecord.query") as mock_query:
        mock_query.side_effect = create_query_side_effect(RECORDS_DIRECT)
        yield mock_query


@pytest.fixture
def mock_calculator_query():
    """Fixture to parameterize CalculatorRecord.query calls."""
    with patch("lambench.metrics.post_process.CalculatorRecord.query") as mock_query:
        mock_query.side_effect = create_query_side_effect(RECORDS_CALCULATOR)
        yield mock_query
