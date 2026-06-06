import gc
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

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
    receraser_unlearning,
    retrain_unlearning,
    rrl_unlearning,
    sisa_unlearning,
    ultrare_unlearning,
    unranking_unlearning,
    unlearnrec_unlearning,
)
from unlearning_metrics import evaluate_forgetting_effect
from utils import get_model

logger = Logger.get_logger("Main")


def compute_edges_to_remove(data, args, device):
    """
    Computes the edges to be removed for the unlearning task.

    Args:
        data: The data object, which includes graph structure and metadata.
        args: Configuration arguments, containing unlearning_task, unlearning_ratio, etc.
        device: The computation device (CPU or CUDA).

    Returns:
        The updated data object with attributes `edges_to_remove` and
        `train_edge_index_after_remove` populated.
    """
    logger.info(
        f"Computing edges to remove for unlearning task: {args['unlearning_task']}"
    )

    train_interaction_data = data.train_edge_index

    if args["unlearning_task"] == "interaction":
        # Interaction unlearning: Remove a portion of recent interactions for a subset of users.
        # 1. Randomly select a subset of users.
        num_users_to_unlearn = int(data.num_users * args["unlearning_ratio"])
        users_to_unlearn = torch.randperm(data.num_users)[:num_users_to_unlearn].to(
            device
        )
        logger.info(
            f"Randomly selected {num_users_to_unlearn} users for interaction unlearning ({args['unlearning_ratio']*100:.1f}%)"
        )

        # 2. For each selected user, identify interactions to unlearn.
        interactions_to_remove_list = []
        # Mask to keep track of edges to be kept.
        mask = torch.ones(
            train_interaction_data.size(1), dtype=torch.bool, device=device
        )

        interaction_ratio = args["unlearning_interaction_ratio"]
        logger.info("Using random selection for interaction unlearning.")

        # Process each user selected for unlearning.
        for user_idx in users_to_unlearn.cpu().numpy():
            # Get all interactions for the current user.
            user_interactions_mask = train_interaction_data[0] == user_idx
            user_interactions = train_interaction_data[:, user_interactions_mask]

            if user_interactions.size(1) == 0:
                continue  # Skip users with no interactions.

            # Randomly select interactions to remove.
            num_interactions_to_remove = max(
                1, int(user_interactions.size(1) * interaction_ratio)
            )
            perm = torch.randperm(user_interactions.size(1), device=device)
            interactions_indices = perm[:num_interactions_to_remove]
            user_interactions_to_remove = user_interactions[:, interactions_indices]

            # Add the interactions to the removal list.
            if user_interactions_to_remove.size(1) > 0:
                interactions_to_remove_list.append(user_interactions_to_remove)
                # Update the global mask to mark interactions for removal.
                for k in range(user_interactions_to_remove.size(1)):
                    interaction_to_mark = user_interactions_to_remove[:, k].unsqueeze(1)
                    # Forward edge match (u -> i)
                    forward_matches = (
                        train_interaction_data == interaction_to_mark
                    ).all(dim=0)
                    # Backward edge match (i -> u)
                    reverse_matches = (
                        train_interaction_data[0] == interaction_to_mark[1]
                    ) & (train_interaction_data[1] == interaction_to_mark[0])
                    # Remove both forward and backward edges.
                    mask[forward_matches | reverse_matches] = False

        # Concatenate all interactions to be removed.
        if interactions_to_remove_list:
            interactions_to_remove = torch.cat(interactions_to_remove_list, dim=1)
        else:
            interactions_to_remove = torch.zeros(
                (2, 0), dtype=torch.long, device=device
            )

        # Update training edge index using the mask.
        new_train_interaction_data = train_interaction_data[:, mask]

        logger.info(
            f"Removed {interactions_to_remove.size(1)} interactions for {num_users_to_unlearn} users."
        )

    elif args["unlearning_task"] == "item":
        # Item unlearning: Randomly remove items and their associated interactions.
        # 1. Randomly select items to remove.
        num_items_to_remove = int(data.num_items * args["unlearning_ratio"])

        # Item indices start after user indices.
        items_to_remove = torch.randperm(data.num_items)[:num_items_to_remove].to(
            device
        )
        # Adjust to get correct item node indices.
        items_to_remove_adjusted = items_to_remove + data.num_users

        # Store information about removed items.
        data.removed_items = items_to_remove_adjusted

        logger.info(
            f"Randomly selected {num_items_to_remove} items for unlearning ({args['unlearning_ratio']*100:.1f}%)"
        )

        # 2. Find all interactions (both directions) related to these items.
        # Reshape for broadcasting.
        items_to_remove_set = items_to_remove_adjusted.view(-1, 1)
        # Forward edges: item is the destination node.
        mask1 = (train_interaction_data[1].view(1, -1) == items_to_remove_set).any(
            dim=0
        )
        # Backward edges: item is the source node.
        mask2 = (train_interaction_data[0].view(1, -1) == items_to_remove_set).any(
            dim=0
        )
        mask = mask1 | mask2

        # 3. Identify interactions to remove and update the training data.
        interactions_to_remove = train_interaction_data[:, mask]
        new_train_interaction_data = train_interaction_data[:, ~mask]

        # Record the node unlearning type as 'item'.
        data.node_type = "item"

        logger.info(
            f"Removed {interactions_to_remove.size(1)} interactions related to {num_items_to_remove} items."
        )

    # Update the data object.
    data.edges_to_remove = interactions_to_remove
    data.train_edge_index_after_remove = new_train_interaction_data

    if hasattr(data, "train_ratings") and not data.train_ratings.empty:
        removed_pairs = set()
        item_lo = data.num_users
        item_hi = data.num_users + data.num_items
        for src, dst in interactions_to_remove.t().detach().cpu().tolist():
            if 0 <= src < data.num_users and item_lo <= dst < item_hi:
                removed_pairs.add((src, dst - data.num_users))
            elif item_lo <= src < item_hi and 0 <= dst < data.num_users:
                removed_pairs.add((dst, src - data.num_users))

        if removed_pairs:
            removed_keys = {u * data.num_items + i for u, i in removed_pairs}
            row_keys = (
                data.train_ratings["user_idx"] * data.num_items
                + data.train_ratings["item_idx"]
            )
            data.train_ratings_after_remove = data.train_ratings[
                ~row_keys.isin(removed_keys)
            ].copy()
        else:
            data.train_ratings_after_remove = data.train_ratings.copy()

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
    
    # Set different seed for each run to ensure randomness
    run_seed = 42 + run_id - 1
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
        min_rating=3,
        min_interactions=10,
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
        f"Average interactions per user: {data.edge_index.size(1) / data.num_users:.2f}"
    )

    # Build a dictionary of user's training items to filter seen items during evaluation
    logger.info("Building user-to-train-items dictionary for evaluation filtering...")
    user_train_items = get_user_train_items(
        data.train_edge_index, data.num_users, data.num_items
    )
    logger.info(
        f"Successfully built user-to-train-items dictionary for {len(user_train_items)} users."
    )

    # Initialize and train the original model
    logger.info("Training the original model...")
    original_model = get_model(
        args["backbone"],
        data.num_users,
        data.num_items,
        emb_dim=args["emb_dim"],
        num_layers=args["num_layers"],
        dropout_rate=args["dropout_rate"],
    ).to(device)

    # Set evaluation batch size based on dataset and available resources
    if args["dataset"] == "ml-1m":
        args["eval_batch_size"] = (
            args["batch_size"] * 2 if torch.cuda.is_available() else 256
        )
    else:  # yelp2018, amazon-book
        args["eval_batch_size"] = (
            args["batch_size"] if torch.cuda.is_available() else 256
        )

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

    logger.info("Original model evaluation results:")
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
        prefix="Before Unlearning",
    )

    # Compute edges to remove
    data = compute_edges_to_remove(data, args, device)

    # Re-compute user-to-train-items dictionary after removal
    user_train_items = get_user_train_items(
        data.train_edge_index_after_remove, data.num_users, data.num_items
    )
    logger.info(
        f"Successfully rebuilt user-to-train-items dictionary for {len(user_train_items)} users."
    )

    # Perform unlearning based on the selected method
    logger.info(f"Performing unlearning using method: {args['method']}")
    unlearning_model, unlearning_time = None, 0.0
    method_name = args["method"].capitalize()

    if args["method"] == "retrain":
        unlearning_model, _, unlearning_time = retrain_unlearning(
            original_model, data, args, device, user_train_items=user_train_items
        )
    elif args["method"] == "gsgcf-ru":
        unlearning_model, _, unlearning_time = gsgcf_ru_unlearning(
            original_model, data, args, device
        )
        method_name = "GSGCF-RU"
    elif args["method"] == "receraser":
        unlearning_model, _, unlearning_time = receraser_unlearning(
            original_model, data, args, device
        )
    elif args["method"] in {"unranking", "raru"}:
        unlearning_model, _, unlearning_time = unranking_unlearning(
            original_model, data, args, device
        )
        method_name = "Unranking"
    elif args["method"] == "ifru":
        unlearning_model, _, unlearning_time = ifru_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "sisa":
        unlearning_model, _, unlearning_time = sisa_unlearning(
            original_model, data, args, device
        )
    elif args["method"] == "certified_removal":
        unlearning_model, _, unlearning_time = certified_removal_unlearning(
            original_model, data, args, device
        )
        method_name = "Certified Removal"
    elif args["method"] == "ultrare":
        unlearning_model, _, unlearning_time = ultrare_unlearning(
            original_model, data, args, device
        )
        method_name = "UltraRE"
    elif args["method"] == "rrl":
        unlearning_model, _, unlearning_time = rrl_unlearning(
            original_model, data, args, device
        )
        method_name = "RRL"
    elif args["method"] == "unlearnrec":
        unlearning_model, _, unlearning_time = unlearnrec_unlearning(
            original_model, data, args, device
        )
        method_name = "UnlearnRec"
    elif args["method"] == "gferaser":
        unlearning_model, _, unlearning_time = gferaser_unlearning(
            original_model, data, args, device
        )
        method_name = "GFEraser"
    else:
        raise ValueError(f"Unknown unlearning method: {args['method']}")

    # Evaluate the unlearned model
    logger.info(f"\nEvaluating the {method_name} model...")
    with torch.no_grad():
        unlearning_model_result = evaluate_model(
            unlearning_model,
            data,
            args,
            device,
            batch_size=args["eval_batch_size"],
            user_train_items=user_train_items,
            is_test=True,
            k=args["k"],
        )

    logger.info(f"\nEvaluation results for {method_name} after unlearning:")
    print_metrics(unlearning_model_result, prefix=f"After {method_name}")
    logger.info(f"\n{method_name} took {unlearning_time:.2f} seconds.")

    # Evaluate forgetting effect
    forgetting_metrics = {}
    if args["evaluate_forgetting"]:
        logger.info(f"\nEvaluating forgetting effect of {method_name}...")
        with torch.no_grad():
            forgetting_metrics = evaluate_forgetting_effect(
                original_model, unlearning_model, data, args, data.edges_to_remove
            )
    else:
        logger.info("Forgetting effect evaluation is disabled.")

    # Perform Membership Inference Attack if enabled
    attack_results = None
    if args["is_attack"]:
        logger.info(
            f"\nPerforming Membership Inference Attack on the model after {method_name}..."
        )
        attack = MembershipInferenceAttack(
            original_model, unlearning_model, data, device, args
        )
        attack_results = attack.run_attack(data.edges_to_remove)

        if attack_results:
            logger.info(
                f"\nDetailed metrics for Membership Inference Attack against {method_name}:"
            )
            logger.info("-" * 50)
            for key, value in attack_results.items():
                if isinstance(value, (list, dict)):
                    logger.info(f"{key}: {value}")
                else:
                    logger.info(f"{key}: {value:.4f}")
            logger.info("-" * 50)

    # Manually clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    total_time = time.time() - start_time
    logger.info(f"\nTotal experiment run time: {total_time:.2f} seconds.")

    # Build results dictionary
    result = {
        "method": args["method"],
        "dataset": args["dataset"],
        "backbone": args["backbone"],
        "test_size": args["test_size"],
        "split_method": args["split_method"],
        "lr": args["lr"],
        "epoch": args["epoch"],
        "batch_size": args["batch_size"],
        "emb_dim": args["emb_dim"],
        "num_layers": args["num_layers"],
        "weight_decay": args["weight_decay"],
        "unlearning_ratio": args["unlearning_ratio"],
        "unlearning_interaction_ratio": args["unlearning_interaction_ratio"],
        "unlearning_task": args["unlearning_task"],
        "influence_hops": args["influence_hops"],
        "unranking_hops": args.get("unranking_hops"),
        "unranking_cg_steps": args.get("unranking_cg_steps"),
        "unranking_damping_lambda": args.get("unranking_damping_lambda"),
        "unranking_dp": args.get("unranking_dp"),
        "unranking_dp_epsilon": args.get("unranking_dp_epsilon"),
        "unranking_dp_delta": args.get("unranking_dp_delta"),
        "iteration": args["iteration"],
        "damp": args["damp"],
        "scale": args["scale"],
        "use_high_impact": args["use_high_impact"],
        "degree_weight": args["degree_weight"],
        "score_weight": args["score_weight"],
        "high_impact_ratio": args["high_impact_ratio"],
        "k": args["k"],
        "recall@k": unlearning_model_result["recall@k"],
        "ndcg@k": unlearning_model_result["ndcg@k"],
        "precision@k": unlearning_model_result["precision@k"],
        "auc": unlearning_model_result["auc"],
        "unlearning_time": unlearning_time,
        "total_time": total_time,
        "URR": forgetting_metrics.get("URR", 0.0),
        "RDR": forgetting_metrics.get("RDR", 0.0),
        "attack_batch_size": args["attack_batch_size"],
        "attack_epochs": args["attack_epochs"],
        "attack_lr": args["attack_lr"],
        "attack_accuracy": attack_results["accuracy"] if attack_results else None,
        "attack_member_success_rate": (
            attack_results["member_success_rate"] if attack_results else None
        ),
        "attack_false_positive_rate": (
            attack_results["false_positive_rate"] if attack_results else None
        ),
        "attack_f1_score": attack_results["f1_score"] if attack_results else None,
        "attack_auc": attack_results["auc"] if attack_results else None,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Note: CSV saving will be handled in run_multiple_experiments function
    
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
        "recall@k", "ndcg@k", "precision@k", "auc", "unlearning_time",
        "total_time", "URR", "RDR", "attack_accuracy", "attack_member_success_rate",
        "attack_false_positive_rate", "attack_f1_score", "attack_auc"
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
    logger.info(f"")
    
    if args['num_runs'] > 1:
        run_multiple_experiments(args)
    else:
        run_single_experiment(args)
