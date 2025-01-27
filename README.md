# LAMBench (in development)

LAMBench is a benchmarking tool designed to evaluate the performance of various machine learning interatomic potential models (MLIPs). It provides a comprehensive suite of tests and metrics to help developers and researchers understand the generalizability of their machine learning models.

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

### Installing with EquiformerV2 models
Using EquiformerV2 models requires the installation of the additional pytorch-geometric packages.
Follow [the instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#additional-libraries), then install `lambench[fairchem]`, e.g.
`pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html`

`mace-torch` pins `e3nn==0.4.4`, which is [not actually required](https://github.com/ACEsuit/mace/issues/555#issuecomment-2423730788) and conflicts with `fairchem`. One can install `lambench[fairchem]` and `lambench[mace]` separately to avoid the conflict.

## Usage

To run the benchmarks, use the following command:

```bash
lambench
```

If there are errors importing the `torch` package regarding symbol error, try:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.1/site-packages/torch/lib/../../nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

## Contributing

We welcome contributions from the community. To contribute, please fork the repository, create a new branch, and submit a pull request with your changes.

### Adding a new model

To add a model, please modify the `lambench/models/models_config.yaml` file.

The file contains a list of models with the following structure:

```yaml
- model_name: a short and concise name for the model
  model_family: the family of the model; used for selecting ASE Calculator in `ase_models.py`
  model_type: select from [ASE, DP]
  model_path: local path to the model weight; null if not required
  virtualenv: (not used yet)
  model_metadata:
    model_description:
  show_direct_task: True
  show_finetune_task: False
  show_calculator_task: False
```

Please refer to `lambench/models/basemodel.py` for the field definitions.

## License

LAMBench is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
