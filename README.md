# Fast Unranking for Preference Revision in Recommendation

## Abstract
Historical interactions can continue to shape recommendations even after the user preferences they reflect have become outdated. Recommendation unlearning provides a practical way to support preference revision by removing the influence of outdated interactions from the trained model. Existing methods typically aim to make the unlearned model behave like one retrained from scratch. However, preference revision requires the model to reflect changes in user preferences while preserving recommendation quality and update efficiency. To address this gap, we propose **Unranking**, a recommendation unlearning framework that revises outdated preferences through targeted ranking updates. Unranking consists of a *Localized Preference Factor* (LPF) and a *Preference-Aware Parameter Update* (PAPU). LPF defines a compact influence scope and assigns a preference factor to each entity in the scope using structural and semantic signals. PAPU uses these factors to demote the target and related items while maintaining recommendation quality on retained interactions. Across five datasets with up to 100 million interactions and three recommendation backbones, Unranking achieves the best retained recommendation quality among unlearning methods in most settings while maintaining high revision effectiveness. It is also the fastest unlearning method in every setting, with a geometric mean speedup of $61\times$ over the unlearning baselines. 

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
