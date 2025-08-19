"""Test for print report generation functionality"""

import tempfile
import json
import pandas as pd
from pathlib import Path

def test_print_report_generation():
    """Test that print report can be generated from sample data"""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample rankings data
        sample_rankings = [
            {
                "Model": "TestModel1",
                "Generalizability-FF Error ↓": 0.25,
                "Generalizability-PC Error ↓": 0.35,
                "Applicability-Instability ↓": 0.10,
                "Applicability-Efficiency ↑": 0.80
            },
            {
                "Model": "TestModel2", 
                "Generalizability-FF Error ↓": 0.30,
                "Generalizability-PC Error ↓": 0.40,
                "Applicability-Instability ↓": 0.15,
                "Applicability-Efficiency ↑": 0.70
            }
        ]
        
        # Create sample barplot data
        sample_barplot = {
            "Molecules": {"TestModel1": 0.20, "TestModel2": 0.25},
            "Inorganic Materials": {"TestModel1": 0.30, "TestModel2": 0.35},
            "Catalysis": {"TestModel1": 0.25, "TestModel2": 0.30}
        }
        
        # Save sample data
        with open(temp_path / "final_rankings.json", "w") as f:
            json.dump(sample_rankings, f, indent=2)
        
        with open(temp_path / "barplot.json", "w") as f:
            json.dump(sample_barplot, f, indent=2)
        
        # Import and test the generate_print_report function
        import sys
        sys.path.insert(0, str(Path.cwd()))
        
        from lambench.metrics.generate_print_report import generate_print_report
        
        # Generate report
        report_path = generate_print_report(temp_path)
        
        # Verify report was created
        assert report_path.exists(), "Report file was not created"
        
        # Read and verify report content
        with open(report_path, "r") as f:
            content = f.read()
        
        # Check for key elements
        assert "LAMBench Performance Report" in content
        assert "TestModel1" in content
        assert "TestModel2" in content
        assert "Molecules" in content
        assert "table" in content
        assert "Executive Summary" in content
        
        print("✓ Print report generation test passed")

if __name__ == "__main__":
    test_print_report_generation()