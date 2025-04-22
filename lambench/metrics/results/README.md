# Overview

Large atomic models (LAM), also known as machine learning interatomic potentials (MLIPs), are considered foundation models that predict atomic interactions across diverse systems using data-driven approaches. LAMBench is a benchmark designed to evaluate the performance of such models. It provides a comprehensive suite of tests and metrics to help developers and researchers understand the accuracy and generalizability of their machine learning models.

## Our mission includes

- **Provide a comprehensive benchmark**: Covering diverse atomic systems across multiple domains, moving beyond domain-specific benchmarks.
- **Align with real-world applications**: Bridging the gap between model performance on benchmarks and their impact on scientific discovery.
- **Enable clear model differentiation**: Offering high discriminative power to distinguish between models with varying performance.
- **Facilitate continuous improvement**: Creating dynamically evolving benchmarks that grow with the community, integrating new tasks and models.

## Features

- **Easy to Use**: Simple setup and configuration to get started quickly.
- **Extensible**: Easily add new benchmarks and metrics.
- **Detailed Reports**: Generates detailed performance reports and visualizations.

# LAMBench Leaderboard
The LAMBench Leaderboard.
$\bar M^m_{\mathrm{FF}}$ refers to the generalizability error on force field prediction tasks, while $\bar M^m_{\mathrm{PC}}$ denotes the generalizability error on domain-specific tasks.
$M_{\mathrm{E}}^m$ stands for the efficiency metric, and $M^m_{\mathrm{IS}}$ refers to the instability metric. Arrows alongside the metrics denote whether a higher or lower value corresponds to better performance.

<!-- radar plot -->
Figure 1: Generalizability on force field prediction tasks, 1 - $\bar{M}^m_{FF}$.
<!-- scatter plot -->
Figure 2: Accuracy-Efficiency Trade-off, $\bar{M}^m_{FF}$ vs $M_E^m$.

# LAMBench Metrics Calculations

## Generalizability

### Force Field Prediction
We categorize all force-field prediction tasks into 5 domains:

- **Inorganic Materials**: `Torres2019Analysis`, `Batzner2022equivariant`, `SubAlex_9k`, `Sours2023Applications`, `Lopanitsyna2023Modeling_A`, `Lopanitsyna2023Modeling_B`, `Dai2024Deep`, `WBM_25k`
- **Small Molecules**: `ANI-1x`
- **Catalysis**: `Vandermause2022Active`, `Zhang2019Bridging`, `Zhang2024Active`, `Villanueva2024Water`
- **Reactions**: `Gasteiger2020Fast`, `Guan2022Benchmark`
- **Biomolecules/Supramolecules**: `MD22`, `AIMD-Chig`

To assess model performance across these domains, we use zero-shot inference with energy-bias term adjustments based on test dataset statistics. Performance metrics are aggregated as follows:

1. The error metric is normalized against the error metric of a baseline model (dummy model) as follows:
$$\hat{M}^m_{k,p,i} = \frac{M^m_{k,p,i}}{M^{\mathrm{dummy}}_{k,p,i}}$$

where $ M^m_{k,p,i} $ is the original error metric, $ m $ indicates the model, $ k $ denotes the domain index, $ p $ signifies the prediction index, and $ i $ represents the test set index.
For instance, in force field tasks, the domains include Small Molecules, Inorganic Materials, Biomolecules, Reactions, and Catalysis, such that $ k \in \{\text{Small Molecules, Inorganic Materials, Biomolecules, Reactions, Catalysis}\} $. The prediction types are categorized as energy ($E$), force ($F$), or virial ($V$), with $ p \in \{E, F, V\} $.
For the specific domain of Reactions, the test sets are indexed as $ i \in \{\text{Guan2022Benchmark, Gasteiger2020Fast}\} $. This baseline model predicts energy based solely on the chemical formula, disregarding any structural details, thereby providing a reference point for evaluating the improvement offered by more sophisticated models.

2. For each domain, we compute the log-average of normalized metrics across all datasets  within this domain by

    $$\bar{M}^m_{k,p} = \exp\left(\frac{1}{n_{k,p}}\sum_{i=1}^{n_{k,p}}\log \hat{M}^m_{k,p,i}\right)$$

where $n_{k,p}$ denotes the number of test sets for domain $k$ and prediction type $p$.

3. Subsequently, we calculate a weighted dimensionless domain error metric to encapsulate the overall error across various prediction types:

    $$\bar{M}^m_{k}  = \sum_p w_{p} \bar{M}^m_{k,p} \Bigg/ \sum_p w_{p}$$

where $ w_{p} $ denotes the weights assigned to each prediction type $p$.

4. Finally the generalizability error metric of a model across all the domains is defined by the average of the domain-wise error metric,

$${\bar M^m}= \frac{1}{n_D}\sum_{k=1}^{n_D}{\bar M^m_{k}}$$

where $n_D$ denotes the number of domains under consideration.

The generalizability error metric $ \bar M^m $ allows for the comparison of generalizability across different models.
It reflects the overall generalization capability across all domains, prediction types, and test sets, with a lower error indicating superior performance.
The only tunable parameter is the weights assigned to prediction types, thereby minimizing arbitrariness in the comparison system.

For the force field generalizability tasks, we adopt RMSE as error metric.
The prediction types include energy and force, with weights assigned as $ w_E = w_F = 0.5 $.
When periodic boundary conditions are assumed and virial labels are available, virial predictions are also considered.
In this scenario, the prediction weights are adjusted to $ w_E = w_F = 0.45 $ and $ w_V = 0.1 $.
The resulting error is referred to as $\bar M^{m}_{FF}$.

The error metric is designed such that a dummy model, which predicts system energy solely based on chemical formulae, results in $\bar{M}^m_{\mathrm{FF}}=1$.
In contrast, an ideal model that perfectly matches Density Functional Theory (DFT) labels achieves a value of $\bar{M}^m_{\mathrm{FF}}=0$.



### Domain Specific Property Calculation

For the domain-specific property tasks, we adopt the MAE as the error metric.
In the Inorganic Materials domain, the MDR phonon benchmark predicts maximum phonon
frequency, entropy, free energy, and heat capacity at constant volume, with each prediction type assigned a weight of 0.25.
In the Small Molecules domain, the TorsionNet500 benchmark predicts the torsion profile energy, torsion barrier height, and the number of molecules for which the model's prediction of the torsional barrier height has an error exceeding 1 kcal/mol.
Each prediction type in this domain is assigned a weight of $\frac{1}{3}$.
The resulting score is denoted as $\bar M^{m}_{PC}$.


## Applicability
### Efficiency

To assess the efficiency of the model, we randomly selected 2000 frames from the domain of Inorganic Materials and Catalysis using the aforementioned out-of-distribution datasets. Each frame was expanded to include 800 to 1000 atoms through the replication of the unit cell, ensuring that measurements of inference efficiency occurred within the regime of convergence. The initial 20% of the test samples were considered a warm-up phase and thus were excluded from the efficiency timing. We have reported the average efficiency across the remaining 1600 frames.

We define an efficiency score,  $M_E^m$, by normalizing the average inference time (with unit $\mathrm{\mu s/atom}$), $\bar \eta^m$, of a given LAM measured over 1600 configurations with respect to an artificial reference value, thereby rescaling it to a range between zero and positive infinity. A larger value indicates higher efficiency.

$$M_E^m = \frac{\eta^0 }{\bar \eta^m },\quad \eta^0= 100\  \mathrm{\mu s/atom}, \quad \bar \eta^m = \frac{1}{1600}\sum_{i}^{1600} \eta_{i}^{m}$$

where $\eta_{i}^{m}$ is the inference time of configuration $i$ for model $m$.

### Stability
Stability is quantified by measuring the total energy drift in NVE simulations across nine structures. For each simulation trajectory, an instability metric is defined based on the magnitude of the slope obtained via linear regression of total energy versus simulation time. This slope is then normalized on a logarithmic scale using a reference value defined as three times the standard deviation of slopes obtained through bootstrap sampling, accounting for numerical error accumulation. The normalized metric is clipped at 0, indicating that drifts smaller than this threshold are not considered significant. If a simulation fails, an instability score of 5 is assigned, representing a drift five orders of magnitude larger than typical numerical error.

$$\tilde{S}_{\Phi_{\mathrm{Drift}, i}}^m =
\begin{cases}
\max\left(0, \log_{10}\frac{\Phi_{\mathrm{Drift}, i}}{\lambda^0}\right), & \text{if success} \\
5, & \text{otherwise}
\end{cases}, \quad
\lambda^0 = 5\times10^{-4} \ \mathrm{eV/atom/ps}$$
where $\Phi_{\mathrm{Drift},i}$ represents the total energy drift of system $i$, and $\lambda ^0$ is the reference.
The final instability metric, $M^m_{\mathrm{IS}}$ is computed as the average over all nine structures.
$$
M^m_{\mathrm{IS}} = \frac{1}{9}\sum_{i=1}^{9}{\tilde{S}_{\Phi_{\mathrm{Drift}, i}}^m}
$$
This result is bounded in the range of [0, $+\infty$], where a score of zero indicates better performance.
