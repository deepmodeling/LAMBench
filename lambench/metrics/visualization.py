import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import lambench
from lambench.metrics.vishelper.results_fetcher import ResultsFetcher
from lambench.metrics.vishelper.metrics_calculations import MetricsCalculator
from lambench.metrics.vishelper.plot_generation import PlotGeneration


class LAMBenchMetrics:
    def __init__(self, timestamp: Optional[datetime] = None):
        self.fetcher = ResultsFetcher(timestamp)
        self.metrics_calculations = MetricsCalculator(self.fetcher)
        self.plot_generation = PlotGeneration(self.fetcher, self.metrics_calculations)

    def save_results(self):
        raw_results = self.fetcher.aggregate_ood_results()
        barplot_data = self.plot_generation.generate_barplot(raw_results)
        radar_chart_config = self.plot_generation.generate_radar_plot(barplot_data)
        scatter_plot_data = self.plot_generation.generate_scatter_plot()
        final_ranking = self.metrics_calculations.summarize_final_rankings()

        result_path = Path(lambench.__file__).parent / "metrics/results"
        with open(result_path / "radar.json", "w") as f:
            json.dump(radar_chart_config, f, indent=2)
            f.write("\n")
        with open(result_path / "scatter.json", "w") as f:
            json.dump(scatter_plot_data, f, indent=2)
            f.write("\n")
        with open(result_path / "barplot.json", "w") as f:
            json.dump(barplot_data, f, indent=2)
            f.write("\n")
        with open(result_path / "final_rankings.json", "w") as f:
            json.dump(final_ranking.to_dict(orient="records"), f, indent=2)
            f.write("\n")
        print("All plots saved to metrics/results/")

    def generate_print_report(self):
        """Generate a comprehensive print-friendly HTML report."""
        # Get all the data needed for the report  
        final_ranking = self.metrics_calculations.summarize_final_rankings()
        
        if final_ranking is None:
            print("Warning: Could not generate rankings. Missing required data.")
            return
        
        # Get barplot data for domain summary
        raw_results = self.fetcher.aggregate_ood_results()
        barplot_data = self.plot_generation.generate_barplot(raw_results)
        
        # Read metadata for descriptions
        result_path = Path(lambench.__file__).parent / "metrics/results"
        try:
            with open(result_path / "metadata.json", "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
        
        # Generate the HTML report
        html_content = self._create_html_report(final_ranking, barplot_data, metadata)
        
        # Save the report
        report_path = result_path / "print_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"Print-friendly report saved to {report_path}")
        return report_path

    def _create_html_report(self, final_ranking, barplot_data, metadata):
        """Create the HTML content for the print report."""
        
        # Get current date for report generation
        generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create rankings table HTML
        rankings_html = self._create_rankings_table(final_ranking)
        
        # Create domain performance summary
        domain_summary_html = self._create_domain_summary(barplot_data)
        
        # Create metric explanations
        metrics_explanation_html = self._create_metrics_explanation()
        
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
        <p>This report presents the performance rankings of {len(final_ranking)} Large Atomistic Models evaluated on the LAMBench benchmark suite. The evaluation covers:</p>
        <ul>
            <li><strong>Generalizability:</strong> Force field prediction accuracy and domain-specific property calculations</li>
            <li><strong>Applicability:</strong> Stability in MD simulations and computational efficiency</li>
        </ul>
        <p><strong>Top Performing Model:</strong> {final_ranking.iloc[0]['Model']} leads the overall rankings with superior performance across multiple metrics.</p>
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

    def _create_rankings_table(self, final_ranking):
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

    def _create_domain_summary(self, barplot_data):
        """Create HTML summary of domain performance."""
        if not barplot_data:
            return '<p>No domain performance data available.</p>'
        
        summary_html = ['<table>']
        summary_html.append('<thead><tr><th>Domain</th><th>Best Model</th><th>Score</th><th>Models Evaluated</th></tr></thead>')
        summary_html.append('<tbody>')
        
        for domain, models in barplot_data.items():
            if models:
                # Find best model (lowest score is better)
                best_model = min(models.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
                model_count = len([v for v in models.values() if v is not None])
                
                summary_html.append('<tr>')
                summary_html.append(f'<td><strong>{domain}</strong></td>')
                summary_html.append(f'<td>{best_model[0]}</td>')
                summary_html.append(f'<td>{best_model[1]:.3f}</td>')
                summary_html.append(f'<td>{model_count}</td>')
                summary_html.append('</tr>')
        
        summary_html.append('</tbody>')
        summary_html.append('</table>')
        
        return '\n'.join(summary_html)

    def _create_metrics_explanation(self):
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
        
        <h4>Property Calculation (PC Error ↓)</h4>
        <p>Evaluates domain-specific property calculations:</p>
        <ul>
            <li><strong>Inorganic Materials:</strong> Phonon properties and elastic constants</li>
            <li><strong>Molecules:</strong> Torsional barriers and conformer energies</li>
            <li><strong>Catalysis:</strong> Energy barriers and reaction energies</li>
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
        <p>Higher values indicate better efficiency.</p>
        
        <h3>Ranking Methodology</h3>
        <p>Models are ranked by sorting on all four metrics with the following priority:</p>
        <ol>
            <li>Generalizability-FF Error (ascending)</li>
            <li>Generalizability-PC Error (ascending)</li>
            <li>Applicability-Instability (ascending)</li>
            <li>Applicability-Efficiency (descending)</li>
        </ol>
        """
        
        return explanation_html


def main(timestamp: Optional[datetime] = None):
    metrics = LAMBenchMetrics(timestamp)
    metrics.save_results()


def generate_print_report(timestamp: Optional[datetime] = None):
    """Generate a print-friendly report from existing data.
    
    This function creates a comprehensive HTML report that includes:
    - Model rankings table with performance metrics
    - Domain performance summary 
    - Detailed metric explanations
    - Print-optimized styling
    
    Args:
        timestamp: Optional timestamp for filtering results
        
    Returns:
        Path to the generated HTML report file
        
    Note:
        This function requires existing JSON result files (final_rankings.json, barplot.json)
        If the full pipeline dependencies are not available, use the standalone
        generate_print_report.py script instead.
    """
    try:
        metrics = LAMBenchMetrics(timestamp)
        return metrics.generate_print_report()
    except Exception as e:
        print(f"Error generating report with full pipeline: {e}")
        print("Try using the standalone script: python lambench/metrics/generate_print_report.py")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--print-report":
        generate_print_report()
    else:
        main()
