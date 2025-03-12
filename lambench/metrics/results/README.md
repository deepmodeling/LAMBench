# Overview

Large atomic models (LAM), also known as machine learning interatomic potentials (MLIPs), are considered fondation models that predict atomic interactions across diverse systems using data-driven approaches. LAMBench is a benchmarking tool designed to evaluate the performance of such models. It provides a comprehensive suite of tests and metrics to help developers and researchers understand the accuracy and generalizability of their machine learning models.

### Our mission includes:

- **Provide a comprehensive benchmark**: Covering diverse atomic systems across multiple domains, moving beyond domain-specific benchmarks.
- **Align with real-world applications**: Bridging the gap between model performance on benchmarks and their impact on scientific discovery.
- **Enable clear model differentiation**: Offering high discriminative power to distinguish between models with varying performance.
- **Facilitate continuous improvement**: Creating dynamically evolving benchmarks that grow with the community, integrating new tasks and models.

### Features
 - **Easy to Use**: Simple setup and configuration to get started quickly.
 - **Extensible**: Easily add new benchmarks and metrics.
 - **Detailed Reports**: Generates detailed performance reports and visualizations.

# LAMBench Leaderboard

### Domain Zero-shot Accuracy

We categorize all zero-shot prediction tasks into 5 domains:
- **Inorganic Materials**: `Torres2019Analysis`, `Batzner2022equivariant`, `SubAlex_9k`, `Sours2023Applications`, `Lopanitsyna2023Modeling_A`, `Lopanitsyna2023Modeling_B`, `Dai2024Deep`, `WBM_25k`
- **Small Molecules**: `ANI-1x`, `Torsionnet500`
- **Catalysis**: `Vandermause2022Active`, `Zhang2019Bridging`, `Zhang2024Active`, `Villanueva2024Water`
- **Reactions**: `Gasteiger2020Fast`, `Guan2022Benchmark`
- **Biomoleculars/Supramoleculars**: `MD22`, `AIMD-Chig`

To assess model performance across these domains, we use zero-shot inference with energy-bias term adjustments based on test dataset statistics. Performance metrics are aggregated as follows:
1. **Metric Normalization**: Each test metric is normalized by its dataset's standard deviation:
    $$\hat{M}_{i,j} = \frac{M_{i,j}}{\sigma_{i,j}}, \quad i \in \{\text{E}, \text{F}, \text{V}\}, \quad j \in \{1,2,\ldots,n\}$$

    where:
    - $\hat{M}_{i,j}$ is the normalized metric for type $i$ on dataset $j$
    - $M_{i,j}$ is the original metric value (mean absolute error)
    - $\sigma_{i,j}$ is the standard deviation of the metric on dataset $j$
    - $i$ denotes the type of metric: E (energy), F (force), V (virial)
    - $j$ indexes over the $n$ datasets in a domain


2. **Domain Aggregation**: For each domain, we compute the log-average of normalized metrics across tasks:
    $$S_i = \exp\left(\frac{1}{n}\sum_{j=1}^{n}\log \hat{M}_{i,j}\right)$$

3. **Combined Score**: We calculate a weighted domain score (lower is better):
    $$S_{\text{domain}} = \begin{cases}
    0.45 \times S_E + 0.45 \times S_F + 0.1 \times S_V & \text{if virial data available} \\
    0.5 \times S_E + 0.5 \times S_F & \text{otherwise}
    \end{cases}$$

4. **Cross-Model Normalization**: We normalize using negative logarithm:
    $$\hat{S}_{\text{domain}} = \frac{-\log(S_{\text{domain}})}{\max_{\text{models}}(-\log(S_{\text{domain}}))}$$

    **Note**: $\hat{S}_{\text{domain}}$ values are displayed on the radar plot.

5. **Overall Performance**: The final model score is the arithmetic mean of all domain scores:
    $$S_{\text{overall}} = \frac{1}{D}\sum_{d=1}^{D} S_{\text{domain}}^d, \quad D=5$$

    **Note**: $S_{\text{overall}}$ values are displayed on the y-axis of the scatter plot.

### Efficiency
