import gc
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from attack import MembershipInferenceAttack
from data_loader import load_dataset, set_seed
from evaluate import evaluate_model, print_metrics
from logger import Logger
from model_cache import ModelCache
from parameters import Config
from trainer import train_model
from unlearning_func import (
    certified_removal_unlearning,
    gferaser_unlearning,
    gsgcf_ru_unlearning,
    ifru_unlearning,
    postfilter_revision,
    receraser_unlearning,
    retrain_unlearning,
    rrl_unlearning,
    sisa_unlearning,
    ultrare_unlearning,
    unranking_revision,
    unlearnrec_unlearning,
)
from unlearning_metrics import evaluate_rdr
from utils import get_model

logger = Logger.get_logger("Main")

_METHOD_DISPLAY = {
    "retrain": "Retrain",
    "postfilter": "Post-filtering",
    "gsgcf-ru": "GSGCF-RU",
    "receraser": "RecEraser",
    "unranking": "Unranking",
    "ifru": "IFRU",
    "sisa": "SISA",
    "certified_removal": "Certified Removal",
    "ultrare": "UltraRE",
    "rrl": "RRL",
    "unlearnrec": "UnlearnRec",
    "gferaser": "GFEraser",
}

# Methods that train/partition the retained set and therefore use D_r at
# evaluation time.  Parameter-update methods keep the original graph.
_RETAINED_GRAPH_METHODS = {
    "retrain",
    "sisa",
    "receraser",
    "ultrare",
    "gsgcf-ru",
    "rrl",
    "unlearnrec",
    "gferaser",
}


def _training_history(data):
    """Return each user's training interactions in chronological order.

    The paper constructs the outdated candidate pool C_u from the earliest
    25% of a user's training interactions, so the history must be ordered by
    timestamp.  If a cache predates the timestamp-aware loader, the stored
    row order is used as a fallback.
    """
    ratings = getattr(data, "train_ratings", None)
    if ratings is not None and not ratings.empty:
        columns = ["user_idx", "item_idx"]
        if "timestamp" in ratings.columns:
            columns.append("timestamp")
        history = ratings[columns].copy()
    else:
        edges = getattr(data, "train_edge_index_original", data.train_edge_index)
        edges = edges[:, (edges[1] >= data.num_users)]
        history = pd.DataFrame(
            {
                "user_idx": edges[0].detach().cpu().numpy(),
                "item_idx": (edges[1] - data.num_users).detach().cpu().numpy(),
            }
        )

    if "timestamp" not in history.columns:
        history["timestamp"] = np.arange(len(history), dtype=np.int64)
    history["timestamp"] = pd.to_numeric(
        history["timestamp"], errors="coerce"
    ).fillna(-1).astype("int64")
    return history.sort_values(
        ["user_idx", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def _build_target_set(data, args):
    """Draw D_t from the outdated candidate pools C_u.

    Interaction level: for every user, sample a revision-ratio fraction of
    the user's interactions uniformly at random from the earliest 25% of the
    training pool.  Entity level: select items and mark all of their
    interactions that fall in the candidate pools, following Section 5.1.4.
    """
    task = str(args.get("revision_level", "interaction")).lower()
    if task == "item":
        task = "entity"
    revision_ratio = float(args.get("revision_ratio", 0.1))
    outdated_fraction = float(args.get("outdated_fraction", 0.25))
    target_seed = int(args.get("target_seed", 42))

    if revision_ratio <= 0:
        return set(), set()
    if not 0 < outdated_fraction <= 1:
        raise ValueError("outdated_fraction must be in (0, 1].")

    history = _training_history(data)
    candidates = {}
    full_lengths = {}
    pool_items = set()
    for user, group in history.groupby("user_idx", sort=False):
        items = group["item_idx"].to_numpy()
        full_lengths[int(user)] = len(items)
        n_candidate = max(1, int(np.floor(outdated_fraction * len(items))))
        n_candidate = min(n_candidate, len(items))
        candidate_items = items[:n_candidate]
        candidates[int(user)] = candidate_items
        pool_items.update(int(item) for item in candidate_items)

    target_pairs = set()
    if task == "interaction":
        for user, candidate_items in tqdm(candidates.items(), desc="Sampling D_t"):
            # The fraction is defined over the user's training interactions,
            # not over the candidate pool.
            full_length = full_lengths[int(user)]
            n_target = int(np.floor(revision_ratio * full_length + 0.5))
            n_target = min(max(n_target, 0), len(candidate_items))
            if n_target <= 0:
                continue
            rng = random.Random(target_seed + int(user))
            chosen = rng.sample(range(len(candidate_items)), n_target)
            for index in chosen:
                target_pairs.add((int(user), int(candidate_items[index])))
        selected_items = set()
    elif task == "entity":
        n_select = int(round(revision_ratio * data.num_items))
        n_select = max(1, min(n_select, len(pool_items))) if pool_items else 0
        rng = random.Random(target_seed + 1)
        selected_items = set(
            rng.sample(sorted(pool_items), n_select)
        ) if n_select > 0 else set()
        for user, candidate_items in candidates.items():
            for item in candidate_items:
                if int(item) in selected_items:
                    target_pairs.add((int(user), int(item)))
    else:
        raise ValueError(
            "revision_level must be 'interaction' or 'entity' "
            "(the legacy name 'item' maps to 'entity')."
        )

    return target_pairs, selected_items


def compute_edges_to_remove(data, args, device):
    """Build the target set D_t and the retained set D_r.

    Returns the data object with ``target_edges`` (the paper's D_t),
    ``edges_to_remove`` (backward-compatible alias), ``retained_edge_index``
    (D_r), and ``train_edge_index_after_remove`` populated.
    """
    logger.info(
        "Drawing the target set D_t from the outdated candidate pools "
        "C_u (task=%s, revision_ratio=%.3f, outdated_fraction=%.2f).",
        args.get("revision_level"),
        float(args.get("revision_ratio", 0.1)),
        float(args.get("outdated_fraction", 0.25)),
    )

    target_pairs, selected_items = _build_target_set(data, args)

    if target_pairs:
        pairs = sorted(target_pairs)
        target_edges = torch.tensor(
            [[u, i + data.num_users] for u, i in pairs],
            dtype=torch.long,
            device=device,
        )
    else:
        target_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        logger.warning("The target set D_t is empty; no revision request is applied.")

    num_items = data.num_items
    target_keys = (
        target_edges[0] * num_items + (target_edges[1] - data.num_users)
        if target_edges.numel() > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )

    def _filter_edges(edge_index):
        if edge_index.numel() == 0:
            return edge_index
        src, dst = edge_index[0], edge_index[1]
        forward = (src >= 0) & (src < data.num_users) & (dst >= data.num_users)
        canonical_user = torch.where(forward, src, dst)
        canonical_item = torch.where(
            forward, dst - data.num_users, src - data.num_users
        )
        keys = canonical_user * num_items + canonical_item
        keep = ~torch.isin(keys, target_keys)
        return edge_index[:, keep]

    data.train_edge_index_after_remove = _filter_edges(data.train_edge_index)
    data.train_edge_index_after_remove_original = _filter_edges(
        getattr(data, "train_edge_index_original", data.train_edge_index)
    )

    if hasattr(data, "train_ratings") and not data.train_ratings.empty:
        ratings = data.train_ratings
        keys = (
            ratings["user_idx"].astype("int64") * num_items
            + ratings["item_idx"].astype("int64")
        )
        keep = ~keys.isin(target_keys.detach().cpu().tolist())
        data.train_ratings_after_remove = ratings[keep].copy()
    else:
        data.train_ratings_after_remove = getattr(data, "train_ratings", None)

    data.target_edges = target_edges
    data.edges_to_remove = target_edges
    data.target_pairs = target_pairs
    data.retained_edge_index = data.train_edge_index_after_remove
    data.pre_revision_edge_index = data.train_edge_index
    data.removed_items = (
        torch.tensor(sorted(selected_items), dtype=torch.long, device=device)
        if selected_items
        else torch.empty(0, dtype=torch.long, device=device)
    )

    num_total = (
        data.train_edge_index_original.size(1)
        if hasattr(data, "train_edge_index_original")
        else data.train_edge_index.size(1)
    )
    logger.info(
        "Target set statistics: |D_t|=%d, |D_r|=%d, revision ratio=%.4f.",
        target_edges.size(1),
        num_total - target_edges.size(1),
        target_edges.size(1) / max(num_total, 1),
    )
    return data


def setup_environment():
    """Sets up the environment, including CUDA and memory settings."""
    # Set CUDA memory allocator configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True"
    )

    # Enable CUDA optimizations
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()

        # Set GPU optimization options
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

        # Enable mixed precision training support
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            logger.info("Mixed precision training supported.")

        # Display GPU information
        gpu_properties = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Model: {gpu_properties.name}")
        logger.info(
            f"Total Memory: {gpu_properties.total_memory / 1024 / 1024 / 1024:.2f} GB"
        )
        logger.info(
            f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB"
        )
        logger.info(
            f"Cached Memory: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.2f} GB"
        )
        logger.info(f"CUDA Version: {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    set_seed(42)

    return device


def get_user_train_items(train_edge_index, num_users, num_items):
    """
    Builds a dictionary mapping users to their training items using vectorized operations.

    Args:
        train_edge_index: Tensor of training edges, shape [2, num_interactions].
        num_users: The total number of users.
        num_items: The total number of items.

    Returns:
        A dictionary where keys are user IDs and values are sets of item IDs they interacted with.
    """
    user_train_items = {}

    # Convert tensors to numpy arrays for faster processing
    users = train_edge_index[0].cpu().numpy()
    # Convert item node indices to item indices
    items = train_edge_index[1].cpu().numpy() - num_users

    # Filter out invalid indices
    valid_mask = (0 <= items) & (items < num_items) & (0 <= users) & (users < num_users)
    users = users[valid_mask]
    items = items[valid_mask]

    # Initialize empty sets for all users
    for u in range(num_users):
        user_train_items[u] = set()

    # Use numpy's efficient indexing to populate the dictionary
    for i in range(len(users)):
        user_idx = users[i]
        item_idx = items[i]
        user_train_items[user_idx].add(item_idx)

    return user_train_items


def run_single_experiment(args, run_id=1):
    """Runs a single experiment."""

    device = setup_environment()
    start_time = time.time()
    
    # Set different seed for each run to ensure randomness.
    run_seed = int(args.get("random_seed", 42)) + run_id - 1
    set_seed(run_seed)
    logger.info(f"Run {run_id} - Using random seed: {run_seed}")

    # Load dataset
    dataset_name = args["dataset"]
    logger.info(f"Training parameters: {args}")
    logger.info(f"\nLoading dataset: {dataset_name}")

    data, _ = load_dataset(
        dataset_name,
        device,
        backbone=args["backbone"],
        min_rating=args["min_rating"],
        core=args["core"],
        test_size=args["test_size"],
        val_size=args["val_size"],
        split_method=args["split_method"],
        use_cache=args["use_cache"],
    )

    # Use original edge count for statistics for fair comparison
    train_edges_cnt = (
        data.train_edge_index_original.size(1)
        if hasattr(data, "train_edge_index_original")
        else data.train_edge_index.size(1)
    )
    test_edges_cnt = (
        data.test_edge_index_original.size(1)
        if hasattr(data, "test_edge_index_original")
        else data.test_edge_index.size(1)
    )
    logger.info(
        f"Dataset stats: {data.num_users} users, {data.num_items} items, "
        f"{train_edges_cnt} training edges, {test_edges_cnt} test edges"
    )
    logger.info(
        f"Average interactions per user: {train_edges_cnt / data.num_users:.2f}"
    )

    # Build a dictionary of user's training items to filter seen items during evaluation
    logger.info("Building user-to-train-items dictionary for evaluation filtering...")
    user_train_items = get_user_train_items(
        getattr(data, "train_edge_index_original", data.train_edge_index),
        data.num_users,
        data.num_items,
    )
    logger.info(
        f"Successfully built user-to-train-items dictionary for {len(user_train_items)} users."
    )

    # Initialize and train the original model
    logger.info("Training the original model...")
    original_model = get_model(args, data.num_users, data.num_items).to(device)

    # Evaluation batch size.  Large datasets keep the batch moderate so the
    # full-item score matrix remains within GPU memory.
    if not torch.cuda.is_available():
        args["eval_batch_size"] = 256
    elif args["dataset"] == "ml-1m":
        args["eval_batch_size"] = args["batch_size"] * 2
    elif args["dataset"] in {"ml-10m", "netflix", "amazon-book"}:
        args["eval_batch_size"] = min(args["batch_size"], 1024)
    else:
        args["eval_batch_size"] = args["batch_size"]

    # Train or load the original model from cache
    if args["use_model_cache"]:
        model_cache = ModelCache()
        logger.info(f"Model cache parameters: {model_cache.get_model_summary(args)}")

        # Check if a cached model exists
        if model_cache.model_exists(args):
            logger.info("Cached model found. Loading from cache...")
            original_model = model_cache.load_model(original_model, args, device)
            logger.info("Model loaded from cache.")
        else:
            logger.info("No cached model found. Starting training...")
            original_model = train_model(
                model=original_model,
                data=data,
                args=args,
                device=device,
                user_train_items=user_train_items,
                eval_interval=args["eval_interval"],
            )
            logger.info("Model training complete. Saving to cache...")
            model_cache.save_model(original_model, args)
            logger.info("Model saved to cache.")
    else:
        logger.info("Model cache disabled. Training model directly...")
        original_model = train_model(
            model=original_model,
            data=data,
            args=args,
            device=device,
            user_train_items=user_train_items,
            eval_interval=args["eval_interval"],
        )

    logger.info("Pre-revision model M_p evaluation results on retained data:")
    print_metrics(
        evaluate_model(
            model=original_model,
            data=data,
            args=args,
            device=device,
            batch_size=args["eval_batch_size"],
            user_train_items=user_train_items,
            is_test=True,
            k=args["k"],
        ),
        prefix="Before Revision (M_p)",
    )

    # Compute edges to remove
    data = compute_edges_to_remove(data, args, device)

    # Re-compute user-to-train-items dictionary after removal
    user_train_items = get_user_train_items(
        getattr(
            data,
            "train_edge_index_after_remove_original",
            data.train_edge_index_after_remove,
        ),
        data.num_users,
        data.num_items,
    )
    logger.info(
        f"Successfully rebuilt user-to-train-items dictionary for {len(user_train_items)} users."
    )

    # Perform revision with the selected method.
    logger.info(f"Performing preference revision using method: {args['method']}")
    revised_model, revision_time = None, 0.0
    method_display = _METHOD_DISPLAY.get(args["method"], args["method"])
    if args["method"] in {"gsgcf-ru", "rrl"} and args["backbone"] != "lightgcn":
        raise ValueError(
            f"{method_display} is tied to the LightGCN/BPR setting in the paper "
            "and is reported only for --backbone lightgcn."
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if args["method"] == "retrain":
        revised_model, _, revision_time = retrain_unlearning(
            original_model, data, args, device, user_train_items=user_train_items
        )
    elif args["method"] == "postfilter":
        revised_model, _, revision_time = postfilter_revision(
            original_model, data, args, device
        )
    elif args["method"] == "gsgcf-ru":
        revised_model, _, revision_time = gsgcf_ru_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "receraser":
        revised_model, _, revision_time = receraser_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "unranking":
        revised_model, _, revision_time = unranking_revision(
            original_model, data, args, device
        )
    elif args["method"] == "ifru":
        revised_model, _, revision_time = ifru_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "sisa":
        revised_model, _, revision_time = sisa_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "certified_removal":
        revised_model, _, revision_time = certified_removal_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "ultrare":
        revised_model, _, revision_time = ultrare_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "rrl":
        revised_model, _, revision_time = rrl_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "unlearnrec":
        revised_model, _, revision_time = unlearnrec_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "gferaser":
        revised_model, _, revision_time = gferaser_unlearning(
            original_model, data, args, device
        )
    else:
        raise ValueError(f"Unknown preference-revision method: {args['method']}")

    # Each method is evaluated on the graph it actually uses: methods that
    # retrain or partition the retained set use D_r, whereas parameter-update
    # methods keep the original graph and change Theta only.
    if args["method"] in _RETAINED_GRAPH_METHODS:
        data.eval_edge_index = data.train_edge_index_after_remove
    else:
        data.eval_edge_index = data.train_edge_index

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    time_memory_product = revision_time * peak_memory_mb / 1024.0  # GB*seconds

    # Evaluate the revised model on retained data (NDCG@10).
    logger.info(f"\nEvaluating the {method_display} model...")
    with torch.no_grad():
        revised_result = evaluate_model(
            revised_model,
            data,
            args,
            device,
            batch_size=args["eval_batch_size"],
            user_train_items=user_train_items,
            is_test=True,
            k=args["k"],
        )

    logger.info(f"\nEvaluation results for {method_display} after revision:")
    print_metrics(revised_result, prefix=f"After {method_display}")
    logger.info(f"\n{method_display} took {revision_time:.2f} seconds.")

    # Ranking-level analysis: Ranking Decrease Rate (RDR).
    rdr_metrics = {}
    if args["evaluate_rdr"]:
        logger.info(f"\nEvaluating the Ranking Decrease Rate of {method_display}...")
        with torch.no_grad():
            rdr_metrics = evaluate_rdr(
                original_model, revised_model, data, args, data.target_edges
            )
    else:
        logger.info("RDR evaluation is disabled.")

    # Distribution-level analysis: Revision Rate (RR) from the MIA probe.
    attack_results = None
    if args["is_attack"]:
        logger.info(
            f"\nRunning the MIA probe for {method_display} "
            "(preference-imprint probe, not a privacy claim)..."
        )
        attack = MembershipInferenceAttack(
            original_model, revised_model, data, device, args
        )
        attack_results = attack.run_attack(data.target_edges)

        if attack_results:
            logger.info("-" * 50)
            logger.info("RR and MIA probe metrics for %s:", method_display)
            for key, value in attack_results.items():
                if isinstance(value, (int, float)) and value is not None:
                    logger.info("  %s: %.4f", key, value)
                else:
                    logger.info("  %s: %s", key, value)
            logger.info("-" * 50)

    # Manually clear memory.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    total_time = time.time() - start_time
    logger.info(f"\nTotal experiment run time: {total_time:.2f} seconds.")

    target_size = int(data.target_edges.size(1))
    retained_size = int(
        getattr(data, "train_edge_index_after_remove_original",
                data.train_edge_index_after_remove).size(1)
    )

    result = {
        "method": args["method"],
        "method_display": method_display,
        "dataset": args["dataset"],
        "backbone": args["backbone"],
        "test_size": args["test_size"],
        "val_size": args["val_size"],
        "split_method": args["split_method"],
        "min_rating": args["min_rating"],
        "core": args["core"],
        "lr": args["lr"],
        "epoch": args["epoch"],
        "batch_size": args["batch_size"],
        "emb_dim": args["emb_dim"],
        "num_layers": args["num_layers"],
        "dropout_rate": args["dropout_rate"],
        "weight_decay": args["weight_decay"],
        "reg_weight": args["reg_weight"],
        "revision_ratio": args["revision_ratio"],
        "revision_level": args["revision_level"],
        "outdated_fraction": args["outdated_fraction"],
        "target_seed": args["target_seed"],
        "target_size": target_size,
        "retained_size": retained_size,
        "target_ratio": target_size / max(target_size + retained_size, 1),
        "eval_graph": (
            "retained" if args["method"] in _RETAINED_GRAPH_METHODS else "original"
        ),
        "influence_hops": args["influence_hops"],
        "unranking_hops": args.get("unranking_hops"),
        "unranking_cg_steps": args.get("unranking_cg_steps"),
        "unranking_damping_lambda": args.get("unranking_damping_lambda"),
        "unranking_no_scoping": args.get("unranking_no_scoping"),
        "unranking_no_preference_factor": args.get("unranking_no_preference_factor"),
        "unranking_no_structural": args.get("unranking_no_structural"),
        "unranking_no_semantic": args.get("unranking_no_semantic"),
        "unranking_no_contextual_correction": args.get(
            "unranking_no_contextual_correction"
        ),
        "unranking_dp": args.get("unranking_dp"),
        "unranking_dp_epsilon": args.get("unranking_dp_epsilon"),
        "unranking_dp_delta": args.get("unranking_dp_delta"),
        "unranking_dp_clip_norm": args.get("unranking_dp_clip_norm"),
        "iteration": args["iteration"],
        "damp": args["damp"],
        "scale": args["scale"],
        "use_high_impact": args["use_high_impact"],
        "degree_weight": args["degree_weight"],
        "score_weight": args["score_weight"],
        "high_impact_ratio": args["high_impact_ratio"],
        "k": args["k"],
        "recall@k": revised_result["recall@k"],
        "ndcg@k": revised_result["ndcg@k"],
        "precision@k": revised_result["precision@k"],
        "auc": revised_result["auc"],
        "revision_time": revision_time,
        "peak_memory_mb": peak_memory_mb,
        "time_memory_product": time_memory_product,
        "total_time": total_time,
        "RR": attack_results.get("RR") if attack_results else None,
        "auroc_pre_revision": (
            attack_results.get("auroc_pre_revision") if attack_results else None
        ),
        "auroc_post_revision": (
            attack_results.get("auroc_post_revision") if attack_results else None
        ),
        "symmetric_auroc_pre_revision": (
            attack_results.get("symmetric_auroc_pre_revision")
            if attack_results
            else None
        ),
        "symmetric_auroc_post_revision": (
            attack_results.get("symmetric_auroc_post_revision")
            if attack_results
            else None
        ),
        "RDR": rdr_metrics.get("RDR", 0.0),
        "rdr_targets": rdr_metrics.get("num_targets", 0),
        "attack_batch_size": args["attack_batch_size"],
        "attack_epochs": args["attack_epochs"],
        "attack_lr": args["attack_lr"],
        "attack_patience": args["attack_patience"],
        "attack_accuracy": attack_results.get("accuracy") if attack_results else None,
        "attack_f1_score": attack_results.get("f1_score") if attack_results else None,
        "attack_auc": attack_results.get("auc") if attack_results else None,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return result


def run_multiple_experiments(args):
    """Runs multiple experiments and computes average results."""
    num_runs = args["num_runs"]
    logger.info(f"Starting {num_runs} experimental runs...")
    
    all_results = []
    
    for run_id in range(1, num_runs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Run {run_id}/{num_runs}")
        logger.info(f"{'='*50}")
        
        # Run single experiment
        result = run_single_experiment(args, run_id)
        all_results.append(result)
        
        logger.info(f"Completed Run {run_id}/{num_runs}")
        
        # Clean up memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate average results
    logger.info(f"\n{'='*50}")
    logger.info(f"Computing average results from {num_runs} runs...")
    logger.info(f"{'='*50}")
    
    # Metrics to average (numeric values only)
    numeric_metrics = [
        "recall@k", "ndcg@k", "precision@k", "auc",
        "revision_time", "peak_memory_mb", "time_memory_product", "total_time",
        "RR", "RDR", "attack_accuracy", "attack_f1_score", "attack_auc",
        "auroc_pre_revision", "auroc_post_revision",
        "symmetric_auroc_pre_revision", "symmetric_auroc_post_revision",
    ]
    
    avg_result = {}
    
    # Copy non-numeric fields from the first result
    for key, value in all_results[0].items():
        if key not in numeric_metrics:
            avg_result[key] = value
    
    # Calculate averages and standard deviations for numeric metrics
    for metric in numeric_metrics:
        values = []
        for result in all_results:
            if result.get(metric) is not None:
                values.append(result[metric])
        
        if values:
            avg_result[metric] = np.mean(values)
            avg_result[f"{metric}_std"] = np.std(values)
            logger.info(f"{metric}: {avg_result[metric]:.4f} ± {avg_result[f'{metric}_std']:.4f}")
        else:
            avg_result[metric] = None
            avg_result[f"{metric}_std"] = None
    
    # Update time with average completion time
    avg_result["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_result["num_runs"] = num_runs
    
    # Save average results to CSV
    result_file = args["output_result_path"]
    if result_file:
        # Create the directory if it does not exist
        result_dir = os.path.dirname(result_file)
        if result_dir and not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # Save individual run results
        individual_results_file = result_file.replace('.csv', '_individual_runs.csv')
        individual_df = pd.DataFrame(all_results)
        individual_df.to_csv(individual_results_file, index=False)
        logger.info(f"Individual run results saved to {individual_results_file}")
        
        # Save average results
        if os.path.exists(result_file):
            try:
                existing_df = pd.read_csv(result_file)
                new_df = pd.DataFrame([avg_result])
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception:
                updated_df = pd.DataFrame([avg_result])
        else:
            updated_df = pd.DataFrame([avg_result])
        
        updated_df.to_csv(result_file, index=False)
        logger.info(f"Average experiment results saved to {result_file}")
    
    logger.info(f"\nCompleted {num_runs} experimental runs successfully!")
    return avg_result, all_results


if __name__ == "__main__":
    parser = Config.get_config()
    args = vars(parser.parse_args())
    logger.info("Command: " + " ".join(sys.argv))
    logger.info("")
    
    if args['num_runs'] > 1:
        run_multiple_experiments(args)
    else:
        run_single_experiment(args)
