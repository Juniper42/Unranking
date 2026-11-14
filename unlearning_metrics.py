import numpy as np
import torch

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
    """Compute scores and full-ranking positions for the target interactions.

    The position ``r_M(u, i)`` is obtained by sorting all items in descending
    order of the model score, as defined in Section 3 of the paper.
    """
    model.eval()
    num_users, num_items = data.num_users, data.num_items

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

    user_to_items = {}
    for u, v in _canonical_user_item_pairs(target_edges, num_users, num_items):
        if 0 <= v < num_items:
            user_to_items.setdefault(u, []).append(v)

    unique_users = sorted(user_to_items.keys())
    scores_dict = {u: {} for u in unique_users}
    rankings_dict = {u: {} for u in unique_users}

    for i in range(0, len(unique_users), batch_size):
        batch_user_ids = unique_users[i : i + batch_size]
        batch_user_indices = torch.tensor(
            batch_user_ids, dtype=torch.long, device=device
        )

        if is_neumf:
            item_indices = torch.arange(num_items, device=device)
            users_expanded = batch_user_indices.view(-1, 1).expand(-1, num_items)
            batch_scores = model(
                users_expanded.flatten(),
                item_indices.repeat(len(batch_user_ids)),
            ).view(len(batch_user_ids), -1)
        else:
            batch_scores = torch.matmul(user_emb[batch_user_indices], item_emb.t())

        batch_rankings = (
            torch.argsort(torch.argsort(batch_scores, dim=1, descending=True), dim=1)
            + 1
        )

        batch_scores_cpu = batch_scores.cpu().numpy()
        batch_rankings_cpu = batch_rankings.cpu().numpy()

        for j, user_id in enumerate(batch_user_ids):
            for item_id in user_to_items[user_id]:
                scores_dict[user_id][item_id] = batch_scores_cpu[j, item_id]
                rankings_dict[user_id][item_id] = batch_rankings_cpu[j, item_id]

    return scores_dict, rankings_dict


def calculate_ranking_decrease_rate(
    original_rankings, revised_rankings, target_edges, data
):
    """Compute the Ranking Decrease Rate (RDR) of Eq. 3.

    ``RDR = (1 / |D_t|) sum_{(u,i) in D_t} (r_M'(u,i) - r_Mp(u,i))``.

    A positive value means that the target interactions were demoted on
    average; ``0`` means that the scores are unchanged.
    """
    rank_changes = []
    for user_id, item_id in _canonical_user_item_pairs(
        target_edges, data.num_users, data.num_items
    ):
        if user_id in original_rankings and user_id in revised_rankings:
            if (
                item_id in original_rankings[user_id]
                and item_id in revised_rankings[user_id]
            ):
                rank_changes.append(
                    revised_rankings[user_id][item_id]
                    - original_rankings[user_id][item_id]
                )

    if not rank_changes:
        return {"RDR": 0.0, "num_targets": 0}

    rank_changes = np.asarray(rank_changes, dtype=np.float64)
    return {
        "RDR": float(rank_changes.mean()),
        "num_targets": int(rank_changes.size),
    }


def evaluate_rdr(original_model, revised_model, data, args, target_edges):
    """Evaluate the Ranking Decrease Rate before and after revision."""
    device = next(original_model.parameters()).device
    logger.info("Evaluating the Ranking Decrease Rate (RDR)...")

    _, original_rankings = compute_scores_and_rankings(
        original_model,
        data,
        args,
        target_edges,
        device,
        graph_edge_index=getattr(data, "pre_revision_edge_index", data.train_edge_index),
    )
    _, revised_rankings = compute_scores_and_rankings(
        revised_model,
        data,
        args,
        target_edges,
        device,
        graph_edge_index=getattr(
            data,
            "eval_edge_index",
            getattr(data, "train_edge_index_after_remove", data.train_edge_index),
        ),
    )

    metrics = calculate_ranking_decrease_rate(
        original_rankings, revised_rankings, target_edges, data
    )
    print_rdr_metrics(metrics)
    return metrics


# Backward-compatible name for older scripts.
evaluate_forgetting_effect = evaluate_rdr


def print_rdr_metrics(metrics):
    logger.info("\n--- Ranking-level Revision Metrics ---")
    logger.info(
        "  RDR (Ranking Decrease Rate): %.4f (Higher is better)",
        metrics.get("RDR", 0.0),
    )
    logger.info("  Targets evaluated: %d", metrics.get("num_targets", 0))
    logger.info("--------------------------------------\n")
