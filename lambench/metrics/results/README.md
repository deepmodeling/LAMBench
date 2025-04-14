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

<!-- radar plot -->
Figure 1: Generalizability score ${S}^m_{k}$ on force field prediction tasks.
<!-- scatter plot -->
Figure 2: Accuracy-Efficiency Trade-off, $\bar{M}^m$ vs $\eta^m$.

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

    $$\bar{M}^m_{k,p} = \exp\left(\frac{1}{n_{k,p}}\sum_{j=1}^{n_{k,p}}\log \hat{M}^m_{k,p,i}\right)$$

where $n_{k,p}$ denotes the number of test sets for domain $k$ and prediction type $p$.

3. Subsequently, we calculate a weighted dimensionless domain error metric to encapsulate the overall error across various prediction types:

    $$\bar{M}^m_{k}  = \sum_p w_{p} \bar{M}^m_{k,p} \Bigg/ \sum_p w_{p}$$

where $ w_{p} $ denotes the weights assigned to each prediction type $p$.

4. The dimensionless domain error metric $ \bar{M}^m_{k} $ is normalized using the negative logarithm to obtain a comparable generalizability score across models:

$${S}^m_{k} = \frac{-\log \bar{M}^m_{k} }{\max_{m}\left(-\log\bar{M}^m_{k}\right)}$$

In this equation, $\max_{m}$ represents the maximum value obtained from all LAM models under comparison.
Thus, a model achieving a score of 1 indicates optimal generalizability among all the models evaluated, whereas the dummy baseline yields a score of 0.

5. Finally the generalizability score of a model across all the domains is defined by the average of the domain scores,

$${S^m}= \frac{1}{n_D}\sum_{i=1}^{n_D}{S^m_{k}}$$

where $n_D$ denotes the number of domains under consideration.

The score $ S^m $ allows for the comparison of generalizability across different models.
It reflects the overall generalization capability across all domains, prediction types, and test sets, with a higher score indicating superior performance.
The only tunable parameter is the weights assigned to prediction types, thereby minimizing arbitrariness in the comparison system.
It is important to note that the score is a relative measure against the best model in the comparison set.
Consequently, the score may change if a better-performing model is introduced into the comparison.

For the force field generalizability tasks, we adopt RMSE as error metric.
The prediction types include energy and force, with weights assigned as $ w_E = w_F = 0.5 $.
When periodic boundary conditions are assumed and virial labels are available, virial predictions are also considered.
In this scenario, the prediction weights are adjusted to $ w_E = w_F = 0.45 $ and $ w_V = 0.1 $.
The resulting score is referred to as $S^{m}_{FF}$.
### Domain Specific Properties

For the domain-specific property tasks, we adopt the MAE as the error metric.
In the Inorganic Materials domain, the MDR phonon benchmark predicts maximum phonon
frequency, entropy, free energy, and heat capacity at constant volume, with each prediction type assigned a weight of 0.25.
In the Small Molecules domain, the TorsionNet500 benchmark predicts the torsion profile energy, torsion barrier height, and the number of molecules for which the model's prediction of the torsional barrier height has an error exceeding 1 kcal/mol.
Each prediction type in this domain is assigned a weight of $\frac{1}{3}$.
The resulting score is denoted as $S^{m}_{DS}$.


## Applicability
### Efficiency

To assess the efficiency of the model, we randomly selected 2000 frames from the domain of Inorganic Materials and Catalysis using the aforementioned out-of-distribution datasets. Each frame was expanded to include 800 to 1000 atoms through the replication of the unit cell, ensuring that measurements of inference efficiency occurred within the regime of convergence. The initial 20% of the test samples were considered a warm-up phase and thus were excluded from the efficiency timing. We have reported the average efficiency across the remaining 1600 frames.

We apply min-max normalization to the average inference efficiency across all LAMs, rescaling it to a range between zero and one. This normalized metric is then subtracted
from one to yield an efficiency score, where a score of one indicates the most efficient model.

$$S_{\eta}^m  = 1 - \frac{\eta^m - \eta_{\min}}{\eta_{\max} - \eta_{\min}}$$

where $ \eta^m $ is the average inference efficiency of the $ m $-th LAM, and $ \eta_{\min} $, $ \eta_{\max} $ are the minimum and maximum efficiency values across all LAMs, respectively.

### Stability
For stability, the same normalization procedure is applied to both the total energy STD and the total energy drift metrics in `NVE Molecular Dynamics`, resulting in two individual stability scores.
$$\tilde{S}_p^m = 1- \frac{S_p^m - S_{\min,p}}{S_{\max,p} - S_{\min,p}} \quad p \in \{{\Phi_{STD}}, {\Phi_{Drift}}\}$$
These are averaged and, if applicable, penalized by multiplying with the success rate to obtain the final stability score.

$$\bar{S^m} = (0.5\times \tilde{S}_{,\Phi_{STD}}^m + 0.5\times \tilde{S}_{\Phi_{Drift}}^m) \times \omega^m$$

The overall applicability score is then calculated as the average of the stability and efficiency scores.

$$S_{A}^m = 0.5\times \bar{S^m} + 0.5\times S_{\eta}^m$$


## Overall Score
The overall score is computed by averaging the two generalizability scores (for force-field and domain-specific tasks) and the applicability score.

$$\text{Overall Score} = \frac{S^{m}_{\text{FF}} + S^{m}_{\text{DS}} + S_A^m}{3}$$
