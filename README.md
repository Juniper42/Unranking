# Fast Unranking for Preference Revision in Recommendation

This repository contains the code for the paper **“Fast Unranking for
Preference Revision in Recommendation”** (TOIS submission).

Unranking is a ranking-aware recommendation unlearning framework for
efficient preference revision.  It reduces the influence of interactions that
no longer reflect a user's current preference while preserving retained
recommendation quality.  The framework has two components:

- **Localized Preference Factor (LPF)**: builds a compact `p`-hop influence
  scope `D_inf` around the target set `D_t` and assigns a preference factor
  `omega(v)` to each entity using structural and semantic signals.
- **Preference-Aware Parameter Update (PAPU)**: builds the aggregated
  demotion gradient `g` and solves
  `(H_inf + lambda I) DeltaTheta = g` with Conjugate Gradient and
  Hessian-vector products, producing `Theta' = Theta + DeltaTheta`.

## Code Structure

```
Unranking/
├── data/                         # datasets (not committed)
├── models/                       # LightGCN, WMF, NeuMF, and the MIA probe
├── unlearning_func/
│   ├── Unranking.py              # LPF + PAPU (Algorithm 1)
│   ├── Retrain.py                # Retrain reference
│   ├── PostFiltering.py          # Post-filtering reference
│   ├── SISA.py
│   ├── CertifiedRemoval.py
│   ├── RecEraser.py
│   ├── UltraRE.py
│   ├── GSGCF_RU.py
│   ├── IFRU.py
│   ├── RRL.py
│   ├── UnlearnRec.py
│   └── GFEraser.py
├── attack.py                     # MIA probe and Revision Rate (RR)
├── data_loader.py                # 10-core filtering, temporal 8:1:1 split
├── evaluate.py                   # retained NDCG@10 / Recall@k
├── unlearning_metrics.py         # Ranking Decrease Rate (RDR)
├── main.py                       # experiment driver and D_t construction
├── parameters.py                 # command-line configuration
├── trainer.py                    # BPR for LightGCN, BCE for WMF/NeuMF
├── model_cache.py
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -n unranking python=3.10
conda activate unranking
pip install -r requirements.txt
```

## Data

Experiments use five datasets: **MovieLens-1M**, **Yelp2018**,
**Amazon-Book**, **MovieLens-10M**, and **Netflix**.  The loader applies the
protocol in Section 5.1.4:

1. ratings of 4 or higher are kept as implicit positives (for files that
   contain ratings);
2. an iterative **10-core** filter is applied to users and items;
3. each user's interactions are split chronologically into 80% training,
   10% validation, and 10% test;
4. the earliest 25% of each user's training interactions form the outdated
   candidate pool `C_u`; a revision-ratio fraction is drawn uniformly at
   random from `C_u` under a fixed seed to form `D_t`.

Expected data layout (one directory per dataset):

```
data/
├── ml-1m/ratings.dat                 # user::item::rating::timestamp
├── ml-10m/ratings.dat                # user::item::rating::timestamp
├── netflix/ratings.csv               # user_id,item_id,rating,timestamp
│   or netflix/combined_data_*.txt
├── yelp2018/ratings.csv              # user_id,item_id,rating,timestamp
│   or yelp2018/total.txt + timestamps.csv
└── amazon-book/ratings.csv
    or amazon-book/total.txt + timestamps.csv
```

`total.txt` is the widely used implicit-feedback release and has no
timestamps.  To reproduce the paper's chronological split for Yelp2018 and
Amazon-Book, provide the recovered interaction times as
`timestamps.csv`/`timestamps.tsv` with columns
`user_id,item_id,timestamp`.  When only `total.txt` is present, the loader
warns and falls back to row order as a timestamp surrogate; this is a
convenience fallback, not the paper protocol.  Yelp `review.json` and Amazon
`reviews*.csv` are also consulted for timestamp recovery when their
identifiers match the preprocessed files.

## Quick Start

Run Unranking with LightGCN on MovieLens-1M under the default 10%
interaction-level revision setting:

```bash
python main.py \
    --method unranking \
    --dataset ml-1m \
    --backbone lightgcn \
    --revision_level interaction \
    --revision_ratio 0.1 \
    --min_rating 4 \
    --split_method temporal \
    --unranking_hops 2 \
    --unranking_cg_steps 30 \
    --unranking_damping_lambda 0.01
```

The paper reports results for `p = 2`; `p = 1` is the lower-latency setting
recommended in Section 5.2.2.  The paper searches `p` in `{0,1,2,3,4}` and
`T` in `{10,20,30,40,50}`.

## Methods

`--method` accepts:

| Family | Methods |
| --- | --- |
| Proposed | `unranking` |
| Reference points | `retrain`, `postfilter` |
| General unlearning | `sisa`, `certified_removal` |
| Recommendation unlearning | `receraser`, `ultrare`, `ifru`, `gsgcf-ru`, `rrl`, `unlearnrec`, `gferaser` |

`gsgcf-ru` and `rrl` are tied to the LightGCN/BPR setting and are reported only
for `--backbone lightgcn`, as in the paper.  The paper evaluates the
LightGCN, WMF, and NeuMF backbones.

## Evaluation

- **Retained recommendation quality**: NDCG@10 (also Recall@k, Precision@k,
  and HR@k) on the retained test interactions.
- **Revision effectiveness**:
  - **Revision Rate (RR)**: ratio of the membership-inference AUC before
    revision (pre-revision model `M_p`) to that after revision (revised model
    `M'`), using the symmetric AUC `max(AUC, 1 - AUC)`.  Each MIA-AUC is
    computed from an MIA probe trained on `M_p` outputs and frozen for `M'`.
  - **Ranking Decrease Rate (RDR)**: mean rank change of `D_t` on the full
    ranking, `RDR = mean(r_M'(u,i) - r_Mp(u,i))`.
- **Revision efficiency**: wall-clock revision time (`Time`) and peak GPU
  memory (`Memory`); the code also reports `time x memory`.

The MIA probe follows Section 5.1.6: members are `D_t`; non-members are
validation/test interactions of the same users matched by item-popularity
group on the retained training data; pairs are split 6:2:2; the probe is a
three-layer MLP with hidden dimensions 64 and 32, trained with BCE and Adam
(`lr = 1e-3`, batch size 256, up to 100 epochs, early stopping after 10 epochs
without validation-AUROC improvement).  RR is a probe for the residual
preference imprint, not a privacy claim.  `--attack_max_pairs` caps the number
of probe pairs for tractability (default 10000; the paper does not specify a
cap).

## Ablations

The flags in Table 6 of the paper are available directly:

```bash
--unranking_no_scoping                 # w/o Scoping
--unranking_no_preference_factor       # w/o Preference Factor
--unranking_no_structural              # w/o Structural
--unranking_no_semantic                # w/o Semantic
--unranking_no_contextual_correction   # w/o Contextual Corr.
--unranking_dp                         # Unranking (+ Noise)
```

The optional noise variant clips `DeltaTheta` to norm `C'`
(`--unranking_dp_clip_norm`) and adds
`N(0, 2 C'^2 ln(1.25/delta) / epsilon^2 I)`; it is off by default.
