# Unranking: A Rank-Aware Recommendation Unlearning Framework for Efficient Preference Revision

## Abstract

Unranking revises outdated preferences in recommender systems by updating model parameters rather than only masking items at the output layer. It first builds a localized influence scope around the target interactions with p-hop propagation and assigns each affected entity a Localized Preference Factor (LPF) from structural frequency and embedding similarity. It then applies a Preference-Aware Parameter Update (PAPU): the target interactions are demoted, and related interactions in the influence scope are demoted in proportion to their preference factors while retained rankings are preserved as much as possible. The optional privacy-oriented variant adds calibrated Gaussian noise to the update.

## Code Structure

```
Unranking/
├── data/
├── models/
├── unlearning_func/
│   ├── Unranking.py
│   ├── Retrain.py
│   ├── SISA.py
│   ├── CertifiedRemoval.py
│   ├── RecEraser.py
│   ├── UltraRE.py
│   ├── GSGCF_RU.py
│   ├── IFRU.py
│   ├── RRL.py
│   ├── UnlearnRec.py
│   └── GFEraser.py
├── attack.py
├── data_loader.py
├── evaluate.py
├── unlearning_metrics.py
├── main.py
├── parameters.py
├── trainer.py
├── requirements.txt
└── README.md
```

## Setup and Installation

1. **Create a conda environment**:
    ```bash
    conda create -n raru python=3.10
    conda activate raru
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

Run the proposed method (Unranking) with LightGCN on ML-1M:

```bash
python main.py \
    --method unranking \
    --dataset ml-1m \
    --backbone lightgcn \
    --unlearning_task interaction \
    --unlearning_ratio 0.1 \
    --unranking_hops 1 \
    --unranking_damping_lambda 0.01 \
    --unranking_cg_steps 30
```

The legacy `--method raru`, `--damping_lambda`, and `--cg_steps` aliases are
still accepted for older scripts, but `--method unranking` and the
`--unranking_*` parameters match the current paper.
