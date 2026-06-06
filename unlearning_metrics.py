import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("UnlearningMetrics")


def _canonical_user_item_pairs(target_edges, num_users, num_items):
    pairs = []
    item_lo = num_users
    item_hi = num_users + num_items
    for src, dst in target_edges.t().detach().cpu().tolist():
        if 0 <= src < num_users and item_lo <= dst < item_hi:
            pairs.append((src, dst - num_users))
        elif item_lo <= src < item_hi and 0 <= dst < num_users:
            pairs.append((dst, src - num_users))
    return pairs


@torch.no_grad()
def compute_scores_and_rankings(
    model, data, args, target_edges, device, batch_size=2048, graph_edge_index=None
):
    """
    Efficiently computes scores, probabilities, and rankings for target edges.
    This function is optimized for performance by processing in batches and minimizing CPU-GPU transfers.

    Args:
        model: The recommendation model.
        data: The data object.
        args: Dictionary of arguments.
        target_edges: The user-item pairs to evaluate.
        device: The computation device.
        batch_size: The batch size for processing.

    Returns:
        A tuple of (scores, rankings) dictionaries for the target edges.
    """
    model.eval()
    num_users, num_items = data.num_users, data.num_items

    # Get model embeddings
    is_gnn = determine_is_gnn(args.get("backbone"))
    backbone = str(args.get("backbone", "")).upper()
    is_neumf = backbone == "NEUMF" and not is_gnn

    if is_gnn:
        if graph_edge_index is None:
            graph_edge_index = getattr(
                data, "train_edge_index_after_remove", data.train_edge_index
            )
        user_emb, item_emb = model(edge_index=graph_edge_index.to(device))
    elif is_neumf:
        user_emb, item_emb = None, None
    else:
        user_emb, item_emb = model.get_embeddings()

    # Create a mapping from user to their target items for efficient lookup
    user_to_items = {}
    for u, v in _canonical_user_item_pairs(target_edges, num_users, num_items):
        if 0 <= v < num_items:
            user_to_items.setdefault(u, []).append(v)

    unique_users = sorted(user_to_items.keys())
    scores_dict = {u: {} for u in unique_users}
    rankings_dict = {u: {} for u in unique_users}

    # Process users in batches
    for i in range(0, len(unique_users), batch_size):
        batch_user_ids = unique_users[i : i + batch_size]
        batch_user_indices = torch.tensor(
            batch_user_ids, dtype=torch.long, device=device
        )

        # Compute scores for all items for the batch of users.
        if is_neumf:
            item_indices = torch.arange(num_items, device=device)
            users_expanded = batch_user_indices.view(-1, 1).expand(-1, num_items)
            batch_scores = model(
                users_expanded.flatten(),
                item_indices.repeat(len(batch_user_ids)),
            ).view(len(batch_user_ids), -1)
        else:
            batch_scores = torch.matmul(user_emb[batch_user_indices], item_emb.t())

        # Compute rankings on the GPU
        # `argsort` twice is a common trick to get ranks: first sorts scores, second gives ranks
        batch_rankings = (
            torch.argsort(torch.argsort(batch_scores, dim=1, descending=True), dim=1)
            + 1
        )

        # Move results to CPU for dictionary population
        batch_scores_cpu = batch_scores.cpu().numpy()
        batch_rankings_cpu = batch_rankings.cpu().numpy()

        for j, user_id in enumerate(batch_user_ids):
            target_item_ids = user_to_items[user_id]
            for item_id in target_item_ids:
                scores_dict[user_id][item_id] = batch_scores_cpu[j, item_id]
                rankings_dict[user_id][item_id] = batch_rankings_cpu[j, item_id]

    return scores_dict, rankings_dict


def calculate_ranking_change(original_rankings, unlearned_rankings, target_edges, data):
    """
    Calculates metrics related to changes in item rankings after unlearning.

    Returns avg_rank_change, rank_decline_ratio, URR, and RDR (Ranking Decrease Rate,
    defined in the Unranking paper as the mean rank shift of target items:
    RDR = (1 / |D_t|) * sum_{(u,i) in D_t} (r_{M'}(u,i) - r_M(u,i)).
    Higher RDR means stronger demotion of target items, which is better for
    rank-aware unlearning.
    """
    rank_changes = []
    urr_terms = []

    for user_id, item_id in _canonical_user_item_pairs(
        target_edges, data.num_users, data.num_items
    ):

        if user_id in original_rankings and user_id in unlearned_rankings:
            if (
                item_id in original_rankings[user_id]
                and item_id in unlearned_rankings[user_id]
            ):
                orig_rank = original_rankings[user_id][item_id]
                unlearn_rank = unlearned_rankings[user_id][item_id]

                # Rank change: positive means rank decreased (good for unlearning)
                rank_change = unlearn_rank - orig_rank
                rank_changes.append(rank_change)

                # Unlearning Rank Rate (URR) term
                urr_terms.append(rank_change / (orig_rank + 1))

    if not rank_changes:
        return {"avg_rank_change": 0.0, "rank_decline_ratio": 0.0,
                "URR": 0.0, "RDR": 0.0}

    rank_changes = np.array(rank_changes)
    urr_terms = np.array(urr_terms)

    # RDR (Ranking Decrease Rate in the Unranking paper) is the
    # un-filtered mean rank shift of target items.
    rdr = float(np.mean(rank_changes))

    # Filter out extreme outliers using the Interquartile Range (IQR) method
    q1, q3 = np.percentile(rank_changes, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    valid_indices = (rank_changes >= lower_bound) & (rank_changes <= upper_bound)

    rank_changes_filtered = rank_changes[valid_indices]
    urr_terms_filtered = urr_terms[valid_indices]

    if rank_changes_filtered.size > 0:
        avg_rank_change = np.mean(rank_changes_filtered)
        # Proportion of items whose rank decreased (moved further down the list)
        rank_decline_ratio = np.mean(rank_changes_filtered > 0)
        urr = np.mean(urr_terms_filtered) * rank_decline_ratio
    else:
        avg_rank_change, rank_decline_ratio, urr = 0.0, 0.0, 0.0

    return {
        "avg_rank_change": avg_rank_change,
        "rank_decline_ratio": rank_decline_ratio,
        "URR": urr,
        "RDR": rdr,
    }


def evaluate_forgetting_effect(
    original_model, unlearned_model, data, args, target_edges
):
    """
    Evaluates the effectiveness of the unlearning process by comparing model outputs
    before and after unlearning.

    Args:
        original_model: The model before unlearning.
        unlearned_model: The model after unlearning.
        data: The data object.
        args: Dictionary of arguments.
        target_edges: The edges that were forgotten.

    Returns:
        A dictionary of forgetting metrics.
    """
    device = next(original_model.parameters()).device
    logger.info("Evaluating unlearning effectiveness...")

    # Compute scores and rankings for the original model
    _, original_rankings = compute_scores_and_rankings(
        original_model,
        data,
        args,
        target_edges,
        device,
        graph_edge_index=data.train_edge_index,
    )

    # Compute scores and rankings for the unlearned model
    _, unlearned_rankings = compute_scores_and_rankings(
        unlearned_model,
        data,
        args,
        target_edges,
        device,
        graph_edge_index=getattr(data, "train_edge_index_after_remove", data.train_edge_index),
    )

    # Calculate metrics based on the changes
    metrics = calculate_ranking_change(
        original_rankings, unlearned_rankings, target_edges, data
    )

    print_forgetting_metrics(metrics)
    return metrics


def print_forgetting_metrics(metrics):
    """Prints the forgetting evaluation metrics in a formatted way."""
    logger.info("\n--- Forgetting Evaluation Metrics ---")
    logger.info(
        f"  Avg. Rank Change:   {metrics.get('avg_rank_change', 0.0):.4f} (Higher is better)"
    )
    logger.info(
        f"  Rank Decline Ratio: {metrics.get('rank_decline_ratio', 0.0):.4f} (Higher is better)"
    )
    logger.info(
        f"  URR (Unlearning Rank Rate): {metrics.get('URR', 0.0):.4f} (Higher is better)"
    )
    logger.info(
        f"  RDR (Ranking Decrease Rate): {metrics.get('RDR', 0.0):.4f} (Higher is better)"
    )
    logger.info("-----------------------------------\n")
