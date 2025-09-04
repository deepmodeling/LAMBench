import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from lambench.workflow.prefect import run_task_prefect, lambench_workflow, submit_tasks_prefect


@pytest.fixture
def mock_task():
    """Create a mock BaseTask instance."""
    mock_task = MagicMock()
    mock_task.task_name = "test_task"
    mock_task.run_task = MagicMock()
    return mock_task


@pytest.fixture
def mock_model():
    """Create a mock BaseLargeAtomModel instance."""
    mock_model = MagicMock()
    mock_model.model_name = "test_model"
    return mock_model


@pytest.fixture
def sample_jobs(mock_task, mock_model):
    """Create sample job list for testing."""
    return [(mock_task, mock_model)]


def test_run_task_prefect_success(mock_task, mock_model):
    """Test successful task execution."""
    # Mock the task execution
    mock_task.run_task.return_value = None
    
    # Execute the Prefect task function directly (not as a Prefect task)
    run_task_prefect.fn(mock_task, mock_model)
    
    # Verify the task was called
    mock_task.run_task.assert_called_once_with(mock_model)


def test_run_task_prefect_failure(mock_task, mock_model):
    """Test task execution failure handling."""
    # Mock the task to raise an exception
    mock_task.run_task.side_effect = ValueError("Test error")
    
    # Execute and expect exception
    with pytest.raises(ValueError, match="Test error"):
        run_task_prefect.fn(mock_task, mock_model)


def test_run_task_prefect_module_not_found(mock_task, mock_model):
    """Test handling of ModuleNotFoundError."""
    # Mock the task to raise ModuleNotFoundError
    mock_task.run_task.side_effect = ModuleNotFoundError("Test module not found")
    
    # Execute and expect exception
    with pytest.raises(ModuleNotFoundError, match="Test module not found"):
        run_task_prefect.fn(mock_task, mock_model)


@patch('lambench.workflow.prefect.run_task_prefect')
def test_lambench_workflow(mock_run_task, sample_jobs):
    """Test the main Prefect workflow."""
    # Mock the task submission and results
    mock_future = MagicMock()
    mock_future.result.return_value = None
    mock_future.task_run.name = "test-task--test-model"
    mock_run_task.with_options.return_value.submit.return_value = mock_future
    
    # Execute the workflow function directly (not as a Prefect flow)
    lambench_workflow.fn(sample_jobs, "test_workflow")
    
    # Verify task was submitted
    mock_run_task.with_options.assert_called_once()
    mock_run_task.with_options.return_value.submit.assert_called_once()
    mock_future.result.assert_called_once()


@patch('lambench.workflow.prefect.lambench_workflow')
def test_submit_tasks_prefect(mock_workflow, sample_jobs):
    """Test the submit_tasks_prefect function."""
    # Execute the submission function
    submit_tasks_prefect(sample_jobs, "test_workflow")
    
    # Verify workflow was executed
    mock_workflow.assert_called_once_with(sample_jobs, "test_workflow")


@patch('lambench.workflow.prefect.lambench_workflow')
def test_submit_tasks_prefect_with_name(mock_workflow, sample_jobs):
    """Test the submit_tasks_prefect function with custom name."""
    # Execute the submission function
    submit_tasks_prefect(sample_jobs, "custom_workflow")
    
    # Verify workflow was executed with custom name
    mock_workflow.assert_called_once_with(sample_jobs, "custom_workflow")


def test_submit_tasks_prefect_empty_jobs():
    """Test submit_tasks_prefect with empty job list."""
    with patch('lambench.workflow.prefect.logging') as mock_logging:
        submit_tasks_prefect([], "test_workflow")
        mock_logging.warning.assert_called_once_with("No jobs to submit to Prefect")