# LOBench: A Benchmark for Limit Order Book Representation Learning

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TBD%202025-blue)](https://ieeexplore.ieee.org/)
[![Pages](https://img.shields.io/badge/Pages-Web%20Overview-orange)](https://pages.muiao.com/pages/lobench/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LOBench** is a standardized benchmark for **Limit Order Book (LOB)** representation learning:

> **Muyao Zhong**, Yushi Lin, and Peng Yang\*, _“Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking,”.

Explore a concise overview of the benchmark and its contributions on the [project webpage](https://pages.muiao.com/pages/lobench/).

---

## Overview

**LOBench** establishes the first **comprehensive and reproducible framework** for LOB representation learning.  
It aims to evaluate and standardize the extraction of transferable, compact features that capture the essential properties of the limit order book — the most fundamental data structure in financial markets.

### Motivation

Existing LOB research often ties deep learning architectures directly to specific downstream tasks (e.g., price prediction), making it difficult to:

- reuse learned representations,
- fairly compare methods,
- or generalize models to new datasets or tasks.

LOBench decouples **representation learning** from **downstream objectives**, enabling a unified and extensible evaluation pipeline.

---

## Key Features

### 1. **Standardized Benchmark**

- Unified data formatting, preprocessing, segmentation, normalization, and labeling.
- Consistent metrics for fair model comparison.

### 2. **Curated Real-World Dataset**

- Derived from **China A-share market (SZSE 2019)**.
- Covers five representative stocks:
  - Ping An Bank (`sz000001`)
  - China Vanke (`sz000002`)
  - Gree Electric (`sz002415`)
  - Wuliangye (`sz000858`)
  - Xiangxue Pharmaceutical (`sz300147`)

### 3. **Diverse Model Zoo**

Includes **9 baseline models** across three categories:

| Type                | Models                            |
| ------------------- | --------------------------------- |
| Foundational        | CNN2, LSTM, Transformer           |
| Generic Time Series | iTransformer, TimesNet, TimeMixer |
| LOB-Specific        | DeepLOB, TransLOB, SimLOB         |

### 4. **Unified Evaluation Framework**

LOBench supports three key downstream tasks:

- **Reconstruction** — Unsupervised evaluation of representational sufficiency
- **Prediction** — Mid-price trend classification
- **Imputation** — Masked LOB recovery

Metrics include **MSE**, **MAE**, **wMSE**, **L<sub>price</sub>**, **L<sub>volume</sub>**, and **L<sub>All</sub>** (structure-regularized loss).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/financial-simulation-lab/LOBench.git
cd LOBench

# Create environment
conda create -n lobench python=3.10
conda activate lobench

# Install dependencies
pip install -r requirements.txt
```

---

### Results Summary

The following table summarizes the best-performing models on each downstream task in LOBench:
| Model | Task | Metric | Best Dataset |
|--------------|---------------|-----------------------|--------------|
| TimesNet | Reconstruction| wMSE = 0.0246 | sz000002 |
| iTransformer | Prediction | CE ↓ | sz000001 |
| SimLOB | Transferability| Recall ↑ = 0.7548 | cross-dataset|

LOBench demonstrates that representation learning can match or surpass task-specific designs while dramatically improving reusability and transferability.

---

### Related Projects

LOBench is part of a broader research ecosystem exploring representation learning and simulation for financial markets.

##### - SimLOB: Representation Learning for Financial Market Simulation

If you wish to learn more about the **SimLOB** model introduced and compared in this paper, please refer to the following publication:

> **Yuanzhe Li**, **Yue Wu**, **Muyao Zhong**, **Shengcai Liu**, and **Peng Yang**.  
> _“SimLOB: Learning Representations of Limited Order Book for Financial Market Simulation.”_ [arXiv:2406.19396](https://arxiv.org/abs/2406.19396)

```bibtex
@misc{li2025simloblearningrepresentationslimited,
      title={SimLOB: Learning Representations of Limited Order Book for Financial Market Simulation},
      author={Yuanzhe Li and Yue Wu and Muyao Zhong and Shengcai Liu and Peng Yang},
      year={2025},
      eprint={2406.19396},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2406.19396},
}
```

##### - Downstream Applications of LOB Representations

For more insights into downstream applications of LOB representation learning, see:

> **Yushi Lin** and **Peng Yang**.  
> _“Detecting Multilevel Manipulation from Limit Order Book via Cascaded Contrastive Representation Learning.”_ [arXiv:2508.17086](https://arxiv.org/abs/2508.17086)

```bibtex
@misc{lin2025detectingmultilevelmanipulationlimit,
      title={Detecting Multilevel Manipulation from Limit Order Book via Cascaded Contrastive Representation Learning},
      author={Yushi Lin and Peng Yang},
      eprint={2508.17086},
      primaryClass={q-fin.CP},
      url={https://arxiv.org/abs/2508.17086}
}
```

### Citation

If you use this benchmark or codebase in your research, please cite:

```bibtex
@misc{zhong2025representationlearninglimitorder,
      title={Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking},
      author={Muyao Zhong and Yushi Lin and Peng Yang},
      year={2025},
      eprint={2505.02139},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2505.02139},
}
```
