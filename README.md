# LAMBench

> [!NOTE]
> Please visit [**the OpenLAM project webpage**](https://www.aissquare.com/openlam?tab=Benchmark) for comprehensive information, interactive results, and community rankings.

## Overview

**LAMBench** is a benchmark designed to evaluate the performance of machine learning interatomic potentials (MLIPs) across multiple domains. It provides a comprehensive suite of tests and metrics to help developers and researchers understand the generalizability of their machine learning models.

Our mission is to:

- **Provide a comprehensive benchmark**: Covering diverse atomic systems across multiple domains, moving beyond domain-specific benchmarks.
- **Align with real-world applications**: Bridging the gap between model performance on benchmarks and their impact on scientific discovery.
- **Enable clear model differentiation**: Offering high discriminative power to distinguish between models with varying performance.
- **Facilitate continuous improvement**: Creating dynamically evolving benchmarks that grow with the community, integrating new tasks and models.

## Features

- **Comprehensive Benchmarks**: Includes a wide range of benchmarks for different downstream tasks.
- **Easy to Use**: Simple setup and configuration to get started quickly.
- **Extensible**: Easily add new benchmarks and metrics.
- **Detailed Reports**: Generates detailed performance reports and visualizations.

## Installation

```bash
pip install git+https://github.com/deepmodeling/LAMBench.git#egg=lambench[deepmd,mace,sevenn,orb]
```

The optional dependencies are required for the corresponding models.

## Usage

To reproduce the results locally or test a custom model, please refer to the `ASEModel.evaluate` method.

- For direct prediction tasks, you can use the staticmethod `run_ase_dptest(calc: Calculator, test_data: Path) -> dict`. The test data can be found [here](https://www.aissquare.com/datasets/detail?pageType=datasets&name=LAMBench-TestData-v1&id=295).
- For calculator tasks, you can use the corresponding scripts provided in `lambench.tasks.calculator`.
  - The phonon test data can be found [here](https://www.aissquare.com/datasets/detail?pageType=datasets&name=LAMBench-Phonon-MDR&id=310).
  - An `ASEModel` object is needed for such tasks; you can create a dummy model as follows:

    ```python
    model = ASEModel(
            model_name="dummy",
            model_type="ASE",
            model_family="<FAMILY_NAME>",
            virtualenv="test",
            model_metadata={
                "test":"test"
            }
        )
    # Note: the corresponding ASE calculator needs to be defined in ASEModel.calc.
    ```
- For finetune tasks, only models based on `DeePMD-kit` framework are supported, please raise an issue if you would like to test other models.

## Contributing

We welcome contributions from the community. To contribute, please fork the repository, create a new branch, and submit a pull request with your changes.

### Adding a new model

To add a model, please modify the `lambench/models/models_config.yaml` file.

The file contains a list of models with the following structure:

  ```yaml
  - model_name: a short and concise name for the model
    model_family: the family of the model; used for selecting ASE Calculator in `ase_models.py`
    model_type: usually `ASE`; use `DP` for deepmd-kit models
    model_path: local path to the model weight; null if not required
    virtualenv: (not used yet)
    model_metadata:
      pretty_name: a human-readable name for the model
      num_parameters: the number of parameters in the model
      model_description:
  ```

Please refer to `lambench/models/basemodel.py` for the field definitions.

Now, add the ASE calculator interface of your model to `lambench/models/ase_models.py`.
Once these modifications are done, please create a pull request. If you have any questions, feel free to create an issue.

### Adding a new task

To add a task (specifically a `calculator` task), please modify the `lambench/tasks/calculator/calculator_tasks.yml` file. Please use [this pull request](https://github.com/deepmodeling/LAMBench/pull/89) as an example.

## License

LAMBench is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
