# Using Prefect Orchestration with LAMBench

LAMBench now supports Prefect as an alternative orchestration backend to dflow. This allows you to use Prefect's powerful workflow management capabilities for running your benchmarking tasks.

## Installation

To use Prefect orchestration, install LAMBench with the Prefect optional dependency:

```bash
pip install lambench[prefect]
```

## Usage

You can use Prefect orchestration by adding the `--prefect` flag to your LAMBench commands:

```bash
# Use Prefect instead of dflow
lambench --prefect --models DP_2024Q4 --tasks HPt_NC_2022

# Compare with dflow (default behavior)
lambench --models DP_2024Q4 --tasks HPt_NC_2022

# Or run locally without any orchestration
lambench --local --models DP_2024Q4 --tasks HPt_NC_2022
```

## Features

The Prefect integration provides:

- **Task Orchestration**: Each model/task combination runs as a separate Prefect task
- **Error Handling**: Robust error handling and logging for failed tasks
- **Monitoring**: Built-in Prefect UI for monitoring workflow execution
- **Scalability**: Ability to scale across multiple workers and environments
- **Retry Logic**: Built-in retry capabilities for transient failures

## Comparison with dflow

| Feature | dflow | Prefect | Local |
|---------|-------|---------|--------|
| Remote Execution | ✅ | ✅ | ❌ |
| Task Monitoring | ✅ | ✅ | ❌ |
| Error Handling | ✅ | ✅ | ✅ |
| Retry Logic | ✅ | ✅ | ❌ |
| UI Dashboard | ✅ | ✅ | ❌ |
| Cloud Integration | Bohrium | Prefect Cloud | ❌ |

## Configuration

The Prefect integration uses the same job gathering logic as the existing dflow implementation:

1. **Models**: Defined in `lambench/models/models_config.yml`
2. **Tasks**: Defined in task configuration files
3. **Execution**: Each (task, model) pair becomes a Prefect task

## Advanced Usage

For production deployments, you may want to:

1. **Set up a Prefect Server**: Run a dedicated Prefect server for better monitoring
2. **Configure Workers**: Set up Prefect workers on different machines
3. **Use Prefect Cloud**: Leverage Prefect's cloud offering for enterprise features

Example with Prefect server:

```bash
# Start Prefect server (in another terminal)
prefect server start

# Run your workflow
lambench --prefect --models DP_2024Q4 --tasks HPt_NC_2022
```

## Implementation Details

The Prefect integration:

- Reuses existing job gathering logic (`gather_jobs`, `gather_models`, `gather_task_type`)
- Maintains the same task execution interface (`task.run_task(model)`)
- Provides equivalent functionality to the dflow implementation
- Includes comprehensive error handling and logging

## Troubleshooting

If you encounter issues:

1. **ImportError**: Ensure Prefect is installed with `pip install lambench[prefect]`
2. **Connection Issues**: Check if you need to configure Prefect server settings
3. **Task Failures**: Check Prefect logs for detailed error information

## Migrating from dflow

To migrate from dflow to Prefect:

1. Install the Prefect dependency
2. Replace `lambench` commands with `lambench --prefect`
3. Optionally set up Prefect server for better monitoring

The same configuration files and task definitions work with both backends.