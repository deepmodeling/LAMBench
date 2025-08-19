#!/usr/bin/env python3
"""
LAMBench Print Report Generator

This script generates a print-friendly HTML report from existing LAMBench results.
It can be used standalone without requiring the full LAMBench environment.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate LAMBench print-friendly report")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="lambench/metrics/results",
        help="Directory containing LAMBench results (default: lambench/metrics/results)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file path (default: results-dir/print_report.html)"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory {results_path} does not exist")
        sys.exit(1)
    
    # Check for required files
    required_files = ["final_rankings.json", "barplot.json"]
    for file in required_files:
        if not (results_path / file).exists():
            print(f"Error: Required file {file} not found in {results_path}")
            sys.exit(1)
    
    try:
        report_path = generate_print_report(results_path, args.output)
        print(f"✓ Print-friendly report generated successfully: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)

def generate_print_report(results_path, output_path=None):
    """Generate a print report using existing data"""
    
    # Load final rankings
    with open(results_path / "final_rankings.json", "r") as f:
        rankings_data = json.load(f)
    
    final_ranking = pd.DataFrame(rankings_data)
    
    # Load barplot data
    with open(results_path / "barplot.json", "r") as f:
        barplot_data = json.load(f)
    
    # Load metadata if available
    metadata_file = results_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Generate HTML report
    html_content = create_html_report(final_ranking, barplot_data, metadata)
    
    # Save report
    if output_path is None:
        output_path = results_path / "print_report.html"
    else:
        output_path = Path(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path

def create_html_report(final_ranking, barplot_data, metadata):
    """Create the HTML content for the print report."""
    
    # Get current date for report generation
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create rankings table HTML
    rankings_html = create_rankings_table(final_ranking)
    
    # Create domain performance summary
    domain_summary_html = create_domain_summary(barplot_data)
    
    # Create metric explanations
    metrics_explanation_html = create_metrics_explanation()
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LAMBench Performance Report</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            line-height: 1.6;
            color: #333;
        }}
        .header {{ 
            text-align: center; 
            border-bottom: 3px solid #2c3e50; 
            padding-bottom: 20px; 
            margin-bottom: 30px;
        }}
        .report-info {{ 
            background-color: #ecf0f1; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 30px;
            border-left: 4px solid #3498db;
        }}
        h1 {{ 
            color: #2c3e50; 
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{ 
            color: #34495e; 
            border-bottom: 2px solid #ecf0f1; 
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 1.8em;
        }}
        h3 {{ 
            color: #34495e; 
            margin-top: 30px;
            font-size: 1.4em;
        }}
        h4 {{ 
            color: #34495e; 
            margin-top: 25px;
            font-size: 1.2em;
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left;
        }}
        th {{ 
            background-color: #34495e; 
            color: white; 
            font-weight: bold;
            text-align: center;
        }}
        tr:nth-child(even) {{ 
            background-color: #f8f9fa; 
        }}
        tr:hover {{ 
            background-color: #e8f4f8; 
        }}
        .rank-1 {{ background-color: #fff3cd !important; font-weight: bold; }}
        .rank-2 {{ background-color: #d1ecf1 !important; }}
        .rank-3 {{ background-color: #d4edda !important; }}
        .metric-lower {{ color: #e74c3c; }}
        .metric-higher {{ color: #27ae60; }}
        .summary-box {{ 
            background-color: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0;
            border-left: 4px solid #27ae60;
        }}
        .formula {{ 
            background-color: #f4f4f4; 
            padding: 10px; 
            border-radius: 4px; 
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            border-left: 3px solid #3498db;
        }}
        .print-only {{ display: none; }}
        @media print {{
            body {{ font-size: 12pt; }}
            .no-print {{ display: none; }}
            .print-only {{ display: block; }}
            table {{ page-break-inside: avoid; }}
            h2 {{ page-break-before: always; }}
            h2:first-of-type {{ page-break-before: avoid; }}
        }}
        .footer {{ 
            margin-top: 50px; 
            text-align: center; 
            color: #7f8c8d; 
            border-top: 1px solid #ecf0f1; 
            padding-top: 20px;
            font-size: 0.9em;
        }}
        .stats {{ 
            display: flex; 
            justify-content: space-around; 
            background-color: #ecf0f1; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0;
        }}
        .stat-item {{ 
            text-align: center; 
        }}
        .stat-number {{ 
            font-size: 2em; 
            font-weight: bold; 
            color: #2c3e50; 
        }}
        .stat-label {{ 
            color: #7f8c8d; 
            font-size: 0.9em; 
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LAMBench Performance Report</h1>
        <p><strong>Large Atomistic Models Benchmark</strong></p>
        <div class="report-info">
            <p><strong>Report Generated:</strong> {generation_date}</p>
            <p><strong>Purpose:</strong> Comprehensive evaluation of Large Atomistic Models (LAMs) across generalizability and applicability metrics</p>
        </div>
    </div>

    <h2>Executive Summary</h2>
    <div class="summary-box">
        <p>This report presents the performance rankings of <strong>{len(final_ranking)} Large Atomistic Models</strong> evaluated on the LAMBench benchmark suite. The evaluation covers:</p>
        <ul>
            <li><strong>Generalizability:</strong> Force field prediction accuracy and domain-specific property calculations</li>
            <li><strong>Applicability:</strong> Stability in MD simulations and computational efficiency</li>
        </ul>
        <p><strong>Top Performing Model:</strong> <em>{final_ranking.iloc[0]['Model']}</em> leads the overall rankings with superior performance across multiple metrics.</p>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(final_ranking)}</div>
                <div class="stat-label">Models Evaluated</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(barplot_data)}</div>
                <div class="stat-label">Domains Tested</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">4</div>
                <div class="stat-label">Core Metrics</div>
            </div>
        </div>
    </div>

    <h2>Model Rankings</h2>
    <p>The following table shows the comprehensive rankings of all evaluated models. Lower values are better for error metrics (↓), while higher values are better for efficiency (↑).</p>
    {rankings_html}

    <h2>Domain Performance Summary</h2>
    <p>Performance breakdown by scientific domain showing generalizability across different types of atomistic systems.</p>
    {domain_summary_html}

    <h2>Metric Definitions and Calculation Methods</h2>
    {metrics_explanation_html}

    <div class="footer">
        <p>Generated by LAMBench - Large Atomistic Models Benchmark Suite</p>
        <p>For more information, visit: <a href="https://github.com/deepmodeling/LAMBench">https://github.com/deepmodeling/LAMBench</a></p>
    </div>
</body>
</html>
    """
    
    return html_template

def create_rankings_table(final_ranking):
    """Create HTML table for model rankings."""
    table_html = ['<table>']
    
    # Header
    headers = final_ranking.columns.tolist()
    table_html.append('<thead><tr>')
    table_html.append('<th>Rank</th>')
    for header in headers:
        # Add styling hints for metric direction
        if '↓' in header:
            table_html.append(f'<th class="metric-lower">{header}</th>')
        elif '↑' in header:
            table_html.append(f'<th class="metric-higher">{header}</th>')
        else:
            table_html.append(f'<th>{header}</th>')
    table_html.append('</tr></thead>')
    
    # Rows
    table_html.append('<tbody>')
    for idx, row in final_ranking.iterrows():
        rank = idx + 1
        rank_class = f'rank-{rank}' if rank <= 3 else ''
        table_html.append(f'<tr class="{rank_class}">')
        table_html.append(f'<td><strong>#{rank}</strong></td>')
        
        for col_name, value in row.items():
            if col_name == 'Model':
                table_html.append(f'<td><strong>{value}</strong></td>')
            else:
                # Format numeric values
                if isinstance(value, (int, float)):
                    table_html.append(f'<td>{value:.3f}</td>')
                else:
                    table_html.append(f'<td>{value}</td>')
        table_html.append('</tr>')
    table_html.append('</tbody>')
    table_html.append('</table>')
    
    return '\n'.join(table_html)

def create_domain_summary(barplot_data):
    """Create HTML summary of domain performance."""
    if not barplot_data:
        return '<p>No domain performance data available.</p>'
    
    summary_html = ['<table>']
    summary_html.append('<thead><tr><th>Domain</th><th>Best Model</th><th>Score</th><th>Models Evaluated</th></tr></thead>')
    summary_html.append('<tbody>')
    
    for domain, models in barplot_data.items():
        if models:
            # Find best model (lowest score is better)
            valid_models = {k: v for k, v in models.items() if v is not None}
            if valid_models:
                best_model = min(valid_models.items(), key=lambda x: x[1])
                model_count = len(valid_models)
                
                summary_html.append('<tr>')
                summary_html.append(f'<td><strong>{domain}</strong></td>')
                summary_html.append(f'<td>{best_model[0]}</td>')
                summary_html.append(f'<td>{best_model[1]:.3f}</td>')
                summary_html.append(f'<td>{model_count}</td>')
                summary_html.append('</tr>')
    
    summary_html.append('</tbody>')
    summary_html.append('</table>')
    
    return '\n'.join(summary_html)

def create_metrics_explanation():
    """Create HTML explanation of metrics."""
    explanation_html = """
    <h3>Generalizability Metrics</h3>
    
    <h4>Force Field Prediction (FF Error ↓)</h4>
    <p>Measures the accuracy of energy, force, and virial predictions across three domains:</p>
    <ul>
        <li><strong>Molecules:</strong> ANI-1x, MD22, AIMD-Chig datasets</li>
        <li><strong>Inorganic Materials:</strong> Torres2019Analysis, Batzner2022equivariant, and others</li>
        <li><strong>Catalysis:</strong> Vandermause2022Active, Zhang2019Bridging, Villanueva2024Water</li>
    </ul>
    <div class="formula">
    Error metric normalized against dummy baseline: M̂ = min(M_model/M_dummy, 1)
    </div>
    <p>The baseline model predicts energy based solely on chemical formula, disregarding structural details.</p>
    
    <h4>Property Calculation (PC Error ↓)</h4>
    <p>Evaluates domain-specific property calculations:</p>
    <ul>
        <li><strong>Inorganic Materials:</strong> Phonon properties (entropy, frequency, heat capacity) and elastic constants</li>
        <li><strong>Molecules:</strong> Torsional barriers (TorsionNet500) and conformer energies (Wiggle150)</li>
        <li><strong>Catalysis:</strong> Energy barriers and reaction energies (OC20NEB-OOD)</li>
    </ul>
    
    <h3>Applicability Metrics</h3>
    
    <h4>Instability (↓)</h4>
    <p>Measures energy drift in NVE molecular dynamics simulations across 9 structures.</p>
    <div class="formula">
    M_IS = max(0, log₁₀(Φ/Φ_tol)) where Φ_tol = 5×10⁻⁴ eV/atom/ps
    </div>
    <p>Lower values indicate better stability. Failed simulations receive a penalty score of 5.</p>
    
    <h4>Efficiency (↑)</h4>
    <p>Computational efficiency measured on 800-1000 atom systems.</p>
    <div class="formula">
    M_E = η₀/η̄ where η₀ = 100 μs/atom (reference), η̄ = average inference time
    </div>
    <p>Higher values indicate better efficiency. Measurements exclude 10% warm-up samples.</p>
    
    <h3>Ranking Methodology</h3>
    <p>Models are ranked by sorting on all four metrics with the following priority:</p>
    <ol>
        <li>Generalizability-FF Error (ascending - lower is better)</li>
        <li>Generalizability-PC Error (ascending - lower is better)</li>
        <li>Applicability-Instability (ascending - lower is better)</li>
        <li>Applicability-Efficiency (descending - higher is better)</li>
    </ol>
    
    <h3>Interpretation Guidelines</h3>
    <ul>
        <li><strong>FF Error = 0:</strong> Perfect match with DFT calculations</li>
        <li><strong>FF Error = 1:</strong> Equivalent to dummy baseline model</li>
        <li><strong>Instability = 0:</strong> Stable MD simulations within tolerance</li>
        <li><strong>Efficiency > 1:</strong> Faster than 100 μs/atom reference</li>
    </ul>
    """
    
    return explanation_html

if __name__ == "__main__":
    main()