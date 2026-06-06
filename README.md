# Unranking: A Rank-Aware Recommendation Unlearning Framework for Efficient Preference Revision

## Abstract

Collaborative filtering encodes historical interactions as positive preference signals. When a user disavows an interaction that no longer reflects a genuine preference, the trained model must revise this outdated signal and reduce its influence. Recommendation unlearning offers a way to perform this revision through parameter updates without full retraining. Many existing methods, however, are privacy-oriented: they prioritize exact removal of requested interactions, often treat targets point by point, and incur costs that preference revision does not always require. We show that an outdated interaction does more than inflate the target item's own score: it shifts the user's implicit preference so that collaboratively or semantically related items are promoted as well. Masking items at the output layer therefore leaves this preference imprint intact, and reducing it requires a rank-aware update of the model parameters. We propose \textbf{\textsc{Unranking}}, a rank-aware recommendation unlearning framework for efficient preference revision. It pairs a \emph{Localized Preference Factor} (LPF), which scopes and weights the affected interactions, with a \emph{Preference-Aware Parameter Update} (PAPU), which demotes them while keeping non-target rankings stable. Across five benchmark datasets and three backbones, \textsc{Unranking} achieves an average 38$\times$ speedup over retraining, maintains recommendation quality on retained data comparable to full retraining, and revises the preference imprint far more thoroughly than output-level post-filtering.

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
