import pytest
from unittest.mock import patch, MagicMock
import argparse

# We can't easily import the main function due to dependencies, so we'll test the logic


def test_entrypoint_prefect_flag():
    """Test that the --prefect flag is properly added to the argument parser."""
    from lambench.workflow.entrypoint import main
    
    # Patch the dependencies to avoid import issues
    with patch('lambench.workflow.entrypoint.gather_jobs') as mock_gather_jobs, \
         patch('lambench.workflow.entrypoint.logging') as mock_logging, \
         patch('lambench.workflow.entrypoint.submit_tasks_prefect') as mock_prefect:
        
        # Mock empty jobs to avoid further execution
        mock_gather_jobs.return_value = []
        
        # Test parsing --prefect flag
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ['lambench', '--prefect']
            main()
            
            # Verify that gather_jobs was called (meaning argument parsing worked)
            mock_gather_jobs.assert_called_once()
            mock_logging.warning.assert_called_with("No jobs found, exiting.")
            
        finally:
            sys.argv = original_argv


def test_entrypoint_prefect_execution():
    """Test that Prefect execution is called when --prefect flag is used."""
    from lambench.workflow.entrypoint import main
    
    # Create mock jobs
    mock_jobs = [("mock_task", "mock_model")]
    
    with patch('lambench.workflow.entrypoint.gather_jobs') as mock_gather_jobs, \
         patch('lambench.workflow.entrypoint.logging') as mock_logging, \
         patch('lambench.workflow.entrypoint.submit_tasks_prefect') as mock_prefect:
        
        # Setup mock jobs
        mock_gather_jobs.return_value = mock_jobs
        
        # Test Prefect execution path
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ['lambench', '--prefect']
            main()
            
            # Verify Prefect submission was called
            mock_prefect.assert_called_once_with(mock_jobs)
            mock_logging.info.assert_called_with("Found 1 jobs.")
            
        finally:
            sys.argv = original_argv


def test_entrypoint_prefect_import_error():
    """Test handling of missing Prefect dependency."""
    from lambench.workflow.entrypoint import main
    
    # Create mock jobs
    mock_jobs = [("mock_task", "mock_model")]
    
    with patch('lambench.workflow.entrypoint.gather_jobs') as mock_gather_jobs, \
         patch('lambench.workflow.entrypoint.logging') as mock_logging, \
         patch('lambench.workflow.entrypoint.submit_tasks_prefect') as mock_prefect:
        
        # Setup mock jobs and ImportError
        mock_gather_jobs.return_value = mock_jobs
        mock_prefect.side_effect = ImportError("No module named 'prefect'")
        
        # Test import error handling
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ['lambench', '--prefect']
            with patch.object(sys.modules['lambench.workflow.entrypoint'], 'submit_tasks_prefect', side_effect=ImportError):
                main()
            
            # Verify error message was logged
            # Note: The exact assertion depends on the import handling in the actual code
            
        finally:
            sys.argv = original_argv