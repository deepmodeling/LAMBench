import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any

from prefect import flow, task


@task
def run_task_prefect(
    task: Any,  # BaseTask
    model: Any,  # BaseLargeAtomModel
) -> None:
    """
    Prefect task that executes a LAMBench task on a model.
    
    This is the Prefect equivalent of dflow's run_task_op function.
    """
    logging.info(f"Running task={task.task_name}, model={model.model_name}")
    try:
        task.run_task(model)
    except ModuleNotFoundError as e:
        logging.error(f"Module not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Task {task.task_name} failed for model {model.model_name}: {e}")
        raise


@flow
def lambench_workflow(jobs: List[Tuple[Any, Any]], name: str = "lambench") -> None:
    """
    Main Prefect flow that orchestrates the execution of LAMBench tasks.
    
    Args:
        jobs: List of (task, model) tuples to execute
        name: Name of the workflow
    """
    logging.info(f"Starting Prefect workflow '{name}' with {len(jobs)} jobs")
    
    # Create a list to collect all task futures
    task_futures = []
    
    for task, model in jobs:
        # Create a unique task name
        task_name = f"{task.task_name}--{model.model_name}"
        # Make task name Prefect-friendly (alphanumeric and hyphens)
        task_name = "".join([c if c.isalnum() else "-" for c in task_name])
        
        # Submit the task for execution
        future = run_task_prefect.with_options(name=task_name).submit(task, model)
        task_futures.append(future)
    
    logging.info(f"Submitted {len(task_futures)} tasks to Prefect")
    
    # Wait for all tasks to complete
    for i, future in enumerate(task_futures):
        try:
            result = future.result()
            task_name = f"Task-{i+1}"  # Simple naming since we can't access task_run.name easily
            logging.info(f"{task_name} completed successfully")
        except Exception as e:
            task_name = f"Task-{i+1}"
            logging.error(f"{task_name} failed: {e}")


def submit_tasks_prefect(
    jobs: List[Tuple[Any, Any]],  # job_list type
    name: str = "lambench",
) -> None:
    """
    Submit tasks for execution using Prefect orchestration.
    
    This function is the Prefect equivalent of submit_tasks_dflow.
    
    Args:
        jobs: List of (task, model) tuples to execute
        name: Name of the workflow
    """
    if not jobs:
        logging.warning("No jobs to submit to Prefect")
        return
    
    logging.info(f"Submitting {len(jobs)} jobs to Prefect workflow")
    
    # For now, we'll run the flow directly
    # In production, you might want to deploy this to a Prefect server
    lambench_workflow(jobs, name)