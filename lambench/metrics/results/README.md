# Overview

Large atomistic models (LAM), also known as machine learning interatomic potentials (MLIPs), are considered foundation models that predict atomic interactions across diverse systems using data-driven approaches. **LAMBench** is a benchmark designed to evaluate the performance of such models. It provides a comprehensive suite of tests and metrics to help developers and researchers understand the accuracy and generalizability of their machine learning models.

## Our mission includes

- **Provide a comprehensive benchmark**: Covering diverse atomic systems across multiple domains, moving beyond domain-specific benchmarks.
- **Align with real-world applications**: Bridging the gap between model performance on benchmarks and their impact on scientific discovery.
- **Enable clear model differentiation**: Offering high discriminative power to distinguish between models with varying performance.
- **Facilitate continuous improvement**: Creating dynamically evolving benchmarks that grow with the community, integrating new tasks and models.

## Features

- **Easy to Use**: Simple setup and configuration to get started quickly.
- **Extensible**: Easily add new benchmarks and metrics.
- **Detailed Reports**: Generates detailed performance reports and visualizations.
- **Print-Friendly Reports**: Creates comprehensive HTML reports suitable for printing and sharing.

## Generating Reports

LAMBench provides multiple ways to generate reports from benchmark results:

### Standard Reports
The standard visualization pipeline generates JSON files with plot configurations:
- `radar.json` - Radar chart configuration
- `scatter.json` - Scatter plot data  
- `barplot.json` - Bar plot data
- `final_rankings.json` - Model rankings table

### Print-Friendly Reports
Generate a comprehensive HTML report suitable for printing and sharing:

```bash
# Using the standalone script (recommended)
python lambench/metrics/generate_print_report.py

# With custom options
python lambench/metrics/generate_print_report.py --results-dir /path/to/results --output report.html

# Using the visualization module (requires full environment)
python lambench/metrics/visualization.py --print-report
```

The print-friendly report includes:
- **Executive Summary**: Key findings and top-performing models
- **Model Rankings**: Comprehensive table with all metrics
- **Domain Performance**: Breakdown by scientific domain
- **Metric Definitions**: Detailed explanations of calculation methods
- **Print Optimization**: CSS styling optimized for printing

# LAMBench Leaderboard

The LAMBench Leaderboard.
$\bar M^m_{\mathrm{FF}}$ refers to the generalizability error on force field prediction tasks, while $\bar M^m_{\mathrm{PC}}$ denotes the generalizability error on domain-specific tasks.
$M_{\mathrm{E}}^m$ stands for the efficiency metric, and $M^m_{\mathrm{IS}}$ refers to the instability metric. Arrows alongside the metrics denote whether a higher or lower value corresponds to better performance.

<!-- radar plot -->
Figure 1: Generalizability on force field prediction tasks, 1 - $\bar{M}^m_{FF}$.
<!-- scatter plot -->
Figure 2: Accuracy-Efficiency Trade-off, $\bar{M}^m_{FF}$ vs $M_E^m$.

# LAMBench Metrics Calculation

## Generalizability

### Force Field Prediction

We categorize all force-field prediction tasks into 3 domains:

- **Inorganic Materials**: `Torres2019Analysis`, `Batzner2022equivariant`, `Sours2023Applications`, `Lopanitsyna2023Modeling`, `Mazitov2024Surface`, `Gao2025Spontaneous`
- **Molecules**: `ANI-1x`, `MD22`, `AIMD-Chig`
- **Catalysis**: `Vandermause2022Active`, `Zhang2019Bridging`, `Villanueva2024Water`

To assess model performance across these domains, we use zero-shot inference with energy-bias term adjustments based on test dataset statistics. Performance metrics are aggregated as follows:

1. The error metric is normalized against the error metric of a baseline model (dummy model) as follows:

    $$\hat{M}^m_{k,p,i} = \min\left(\frac{M^m_{k,p,i}}{M^{\mathrm{dummy}}_{k,p,i}},\quad 1\right)$$

    where $M^m_{k,p,i}$ is the original error metric, $m$ indicates the model, $k$ denotes the domain index, $p$ signifies the prediction index, and $i$ represents the test set index. For a model with worse accuracy than a dummy model, the error metric is set to 1.
    For instance, in force field tasks, the domains include Molecules, Inorganic Materials, and Catalysis, such that $k \in \{\text{Molecules, Inorganic Materials, Catalysis}\}$. The prediction types are categorized as energy ($E$), force ($F$), or virial ($V$), with $p \in \{E, F, V\}$.
    For the specific domain of Molecules, the test sets are indexed as $i \in \{\text{ANI-1x, MD22, AIMD-Chig}\}$. This baseline model predicts energy based solely on the chemical formula, disregarding any structural details, thereby providing a reference point for evaluating the improvement offered by more sophisticated models.

2. For each domain, we compute the log-average of normalized metrics across all datasets  within this domain by

    $$\bar{M}^m_{k,p} = \exp\left(\frac{1}{n_{k,p}}\sum_{i=1}^{n_{k,p}}\log \hat{M}^m_{k,p,i}\right)$$

    where $n_{k,p}$ denotes the number of test sets for domain $k$ and prediction type $p$.

3. Subsequently, we calculate a weighted dimensionless domain error metric to encapsulate the overall error across various prediction types:

    $$\bar{M}^m_{k}  = \sum_p w_{p} \bar{M}^m_{k,p} \Bigg/ \sum_p w_{p}$$

    where $w_{p}$ denotes the weights assigned to each prediction type $p$.

4. Finally the generalizability error metric of a model across all the domains is defined by the average of the domain-wise error metric,

    $${\bar M^m}= \frac{1}{n_D}\sum_{k=1}^{n_D}{\bar M^m_{k}}$$

    where $n_D$ denotes the number of domains under consideration.

The generalizability error metric $\bar M^m$ allows for the comparison of generalizability across different models.
It reflects the overall generalization capability across all domains, prediction types, and test sets, with a lower error indicating superior performance.
The only tunable parameter is the weights assigned to prediction types, thereby minimizing arbitrariness in the comparison system.

For the force field generalizability tasks, we adopt RMSE as error metric.
The prediction types include energy and force, with weights assigned as $w_E = w_F = 0.5$.
When periodic boundary conditions are assumed and virial labels are available, virial predictions are also considered.
In this scenario, the prediction weights are adjusted to $w_E = w_F = 0.45$ and $w_V = 0.1$.
The resulting error is referred to as $\bar M^{m}_{FF}$.

The error metric is designed such that a dummy model, which predicts system energy solely based on chemical formulae, results in $\bar{M}^m_{\mathrm{FF}}=1$.
In contrast, an ideal model that perfectly matches Density Functional Theory (DFT) labels achieves a value of $\bar{M}^m_{\mathrm{FF}}=0$.

### Domain Specific Property Calculation

For the domain-specific property calculation tasks, we adopt the MAE as the primary error metric.

In the Inorganic Materials domain, the MDR phonon benchmark predicts the maximum phonon frequency, entropy, free energy, and heat capacity at constant volume, while the elasticity benchmark evaluates the shear and bulk moduli. Each prediction type
is assigned an equal weight of $\frac{1}{6}$.

In the Molecules domain, the TorsionNet500 benchmark evaluates the torsion profile energy, torsional barrier height, and the number of molecules for which the predicted torsional barrier height error exceeds 1 kcal/mol. The Wiggle150 benchmark assesses the relative conformer energy profile. Each prediction type in this domain is assigned a weight of 0.25.

In the Catalysis domain, the OC20NEB-OOD benchmark evaluates the energy barrier, reaction energy change (delta energy), and the percentage of reactions with predicted energy barrier errors exceeding 0.1 eV for three reaction types: transfer, dissociation, and desorption. Each prediction type in this domain is assigned a weight of 0.2.

The resulting error metric after averaging over all domains is denoted as $\bar M^{m}_{PC}$.

## Applicability

### Efficiency

To assess the efficiency of the model, we randomly selected 1000 frames from the domain of Inorganic Materials and Catalysis using the aforementioned out-of-distribution datasets. Each frame was expanded to contain between 800 and 1000 atoms — dynamically determined using a binary search algorithm to fully utilize GPU capacity — by replicating the unit cell. This ensured that measurements of inference efficiency were conducted within the regime of convergence. The initial 10% of the test samples were considered a warm-up phase and thus were excluded from the efficiency timing. We have reported the average efficiency across the remaining 900 frames.

We define an efficiency score,  $M_E^m$, by normalizing the average inference time (with unit $\mathrm{\mu s/atom}$), $\bar \eta^m$, of a given LAM measured over 900 configurations with respect to an artificial reference value, thereby rescaling it to a range between zero and positive infinity. A larger value indicates higher efficiency.

$$M_E^m = \frac{\eta^0 }{\bar \eta^m },\quad \eta^0= 100\  \mathrm{\mu s/atom}, \quad \bar \eta^m = \frac{1}{900}\sum_{i}^{900} \eta_{i}^{m}$$

where $\eta_{i}^{m}$ is the inference time of configuration $i$ for model $m$.

### Stability

Stability is quantified by measuring the total energy drift in NVE simulations across nine structures.
For each simulation trajectory, an instability metric is defined based on the magnitude of the slope obtained via linear regression of total energy per atom versus simulation time. A tolerance value, $5\times10^{-4} \ \mathrm{eV/atom/ps}$,  is determined as three times the statistical uncertainty in calculating the slope from a 10 ps NVE-MD trajectory using the MACE-MPA-0 model. If the measured slope is smaller than the tolerance value, the energy drift is considered negligible. We define the dimensionless measure of instability for structure $i$ as follows:

If the computation is successful:

$$M^m_{\mathrm{IS},i} = \max\left(0, \log_{10}\left(\frac{\Phi_{i}}{\Phi_{\mathrm{tol}}}\right)\right)$$

Otherwise:

$$M^m_{\mathrm{IS},i} = 5$$

with
$$\Phi_{\mathrm{tol}} = 5 \times 10^{-4} \ \mathrm{eV/atom/ps}$$
where $\Phi_i$ represents the total energy drift , and $\Phi_{\mathrm{tol}}$ denotes the tolerance.
This metric indicates the relative order of magnitude of the slope compared to the tolerance.
In cases where a MD simulation fails, a penalty of 5 is assigned, representing a drift five orders of magnitude larger than the typical statistical uncertainty in measuring the slope.
The final instability metric is computed as the average over all nine structures.

$$M^m_{\mathrm{IS}} = \frac{1}{9}\sum_{i=1}^{9} M^m_{\mathrm{IS},i}$$

This result is bounded within the range $[0, +\infty)$, where a lower value signifies greater stability.
