import numpy as np
import torch

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Evaluation")


@torch.no_grad()
def evaluate_model(
    model,
    data,
    args,
    device,
    batch_size=4096,
    k=10,
    user_train_items=None,
    is_test=True,
):
    """
    Evaluates the model's performance.

    Args:
        model: The model to evaluate.
        data: The data object.
        args: Configuration arguments, including 'backbone'.
        device: The computation device.
        batch_size: The batch size for evaluation.
        k: The number of top recommendations to consider.
        user_train_items: A dictionary mapping user indices to their training items,
                          used to filter out seen items during evaluation.
        is_test: Flag to indicate whether to use the test set (True) or validation set (False).

    Returns:
        A dictionary containing performance metrics (Recall@k, NDCG@k, etc.).
    """
    model.eval()
    model = model.to(device)

    if hasattr(model, "clear_cache"):
        model.clear_cache()

    if is_test:
        eval_edge_index_orig = getattr(
            data, "test_edge_index_original", data.test_edge_index
        )
    else:
        eval_edge_index_orig = getattr(data, "val_edge_index_original", None)
        if eval_edge_index_orig is None:
            eval_edge_index_orig = getattr(data, "val_edge_index", None)
        if eval_edge_index_orig is None or eval_edge_index_orig.size(1) == 0:
            logger.warning(
                "Validation set is empty or does not exist. Skipping evaluation."
            )
            return {
                "recall@k": 0.0,
                "ndcg@k": 0.0,
                "precision@k": 0.0,
                "f1@k": 0.0,
                "auc": 0.0,
                "hr@k": 0.0,
            }

    eval_edge_index = eval_edge_index_orig.to(device)
    
    backbone = args["backbone"].upper()
    is_gnn_model = determine_is_gnn(backbone)
    is_neumf = backbone == "NEUMF" and not is_gnn_model

    # Post-filtering leaves the model unchanged but masks outdated items in
    # the output list (paper Section 5.1.3).  Interaction-level targets are
    # masked per affected user so that one outdated interaction does not
    # remove the item globally; entity-level targets are whole outdated items
    # and are masked for every user.  RDR is computed separately on the full,
    # unmasked ranking, matching Section 3's rank definition.
    postfilter_global_items = None
    postfilter_users = None
    postfilter_items = None
    if str(args.get("method", "")).lower() == "postfilter":
        target_edges = getattr(data, "target_edges", None)
        if target_edges is not None and target_edges.numel() > 0:
            edges = target_edges.to(device)
            items = edges[1] - data.num_users
            valid = (items >= 0) & (items < data.num_items)
            if bool(valid.any()):
                level = str(args.get("revision_level", "interaction")).lower()
                if level in {"entity", "item"}:
                    # Entity-level targets are whole items, so mask them for
                    # every user, not only for the users in the candidate pool.
                    postfilter_global_items = torch.unique(items[valid])
                else:
                    users = edges[0][valid]
                    items = items[valid]
                    order = torch.argsort(users)
                    postfilter_users = users[order]
                    postfilter_items = items[order]

    if is_gnn_model:
        graph_edge_index = getattr(
            data,
            "eval_edge_index",
            getattr(data, "train_edge_index_after_remove", data.train_edge_index),
        )
        user_emb, item_emb = model(edge_index=graph_edge_index.to(device))
    elif is_neumf:
        user_emb, item_emb = None, None # NeuMF scores all candidates, no pre-computed embeddings needed
    else:
        user_emb, item_emb = model.get_embeddings()

    # Sparse per-user masks of seen training items.  A dense
    # num_users x num_items mask is infeasible on Netflix
    # (480,189 x 177,770), so the mask is applied row by row.
    train_items_by_user = {}
    if user_train_items is not None:
        for user_id, item_ids in user_train_items.items():
            if 0 <= user_id < data.num_users and item_ids:
                items = torch.tensor(
                    sorted(item_ids), dtype=torch.long, device=device
                )
                items = items[(0 <= items) & (items < data.num_items)]
                train_items_by_user[user_id] = items
    elif hasattr(data, "train_ratings") and not data.train_ratings.empty:
        grouped = data.train_ratings.groupby("user_idx")["item_idx"].apply(list)
        for user_id, item_ids in grouped.items():
            train_items_by_user[int(user_id)] = torch.tensor(
                sorted(item_ids), dtype=torch.long, device=device
            )

    # Prepare ground truth dictionary for evaluation users
    eval_users = eval_edge_index[0]
    eval_items = eval_edge_index[1] - data.num_users # Adjust to item-specific indices
    
    ground_truth = {}
    for i in range(len(eval_users)):
        user = eval_users[i].item()
        item = eval_items[i].item()
        if user not in ground_truth:
            ground_truth[user] = []
        ground_truth[user].append(item)
        
    unique_users_tensor = torch.tensor(list(ground_truth.keys()), device=device)
    test_items_list_gpu = [torch.tensor(ground_truth[u], device=device) for u in unique_users_tensor.tolist()]

    num_eval_users = unique_users_tensor.size(0)
    num_batches = (num_eval_users + batch_size - 1) // batch_size
    recalls, ndcgs, precisions, f1_scores, auc_scores, hit_rates = [], [], [], [], [], []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_eval_users)
        batch_user_ids = unique_users_tensor[start:end]
        
        if is_neumf:
            # Score all items for the batch of users
            item_indices = torch.arange(data.num_items, device=device)
            users_expanded = batch_user_ids.view(-1, 1).expand(-1, data.num_items)
            scores_batch = model(users_expanded.flatten(), item_indices.repeat(len(batch_user_ids))).view(len(batch_user_ids), -1)
        else:
            batch_user_emb = user_emb[batch_user_ids]
            scores_batch = torch.matmul(batch_user_emb, item_emb.t())

        # Apply the sparse seen-item mask.
        for row, user_id in enumerate(batch_user_ids.tolist()):
            seen = train_items_by_user.get(user_id)
            if seen is not None and seen.numel() > 0:
                scores_batch[row, seen] = float("-inf")

        # Post-filtering masks the outdated items of each affected user.
        if postfilter_global_items is not None and postfilter_global_items.numel() > 0:
            scores_batch[:, postfilter_global_items] = float("-inf")
        if postfilter_users is not None:
            lo = int(torch.searchsorted(postfilter_users, batch_user_ids.min().item()))
            hi = int(
                torch.searchsorted(postfilter_users, batch_user_ids.max().item() + 1)
            )
            users_in_range = postfilter_users[lo:hi]
            items_in_range = postfilter_items[lo:hi]
            if users_in_range.numel() > 0:
                sorted_batch, _ = torch.sort(batch_user_ids)
                positions = torch.searchsorted(sorted_batch, users_in_range)
                hit = (positions < sorted_batch.numel()) & (
                    sorted_batch[positions.clamp(max=sorted_batch.numel() - 1)]
                    == users_in_range
                )
                if bool(hit.any()):
                    scores_batch[positions[hit], items_in_range[hit]] = float("-inf")

        _, topk_indices = torch.topk(scores_batch, k=k, dim=1)
        
        # Vectorized metrics calculation
        batch_gt_items = test_items_list_gpu[start:end]
        
        # Create a boolean matrix indicating hits
        # Unsqueeze to enable broadcasting for comparison
        hits = torch.stack(
            [
                torch.isin(topk_indices[row], gt_items)
                for row, gt_items in enumerate(batch_gt_items)
            ],
            dim=0,
        )

        pos_lens = torch.tensor([len(gt) for gt in batch_gt_items], device=device)
        hit_counts = hits.sum(dim=1)
        
        recall_batch = hit_counts / pos_lens.float().clamp(min=1)
        precision_batch = hit_counts / k
        hr_batch = hit_counts > 0
        
        f1_batch = torch.where(
            (recall_batch + precision_batch) > 0,
            2 * recall_batch * precision_batch / (recall_batch + precision_batch),
            torch.zeros_like(recall_batch)
        )
        
        # Calculate NDCG
        positions = torch.arange(1, k + 1, device=device).float()
        discounts = 1.0 / torch.log2(positions + 1)
        dcg = (hits * discounts).sum(dim=1)
        idcg_vals = torch.tensor([torch.sum(discounts[:min(k, pl)]) for pl in pos_lens], device=device)
        ndcg_batch = dcg / idcg_vals.clamp(min=1e-6)

        recalls.extend(recall_batch.tolist())
        precisions.extend(precision_batch.tolist())
        hit_rates.extend(hr_batch.tolist())
        f1_scores.extend(f1_batch.tolist())
        ndcgs.extend(ndcg_batch.tolist())
        
        # AUC is calculated per user
        for i, user_id in enumerate(batch_user_ids):
            pos_items = ground_truth[user_id.item()]
            user_scores = scores_batch[i]
            pos_scores = user_scores[pos_items]
            all_neg_items = torch.ones(
                data.num_items, dtype=torch.bool, device=device
            )
            seen = train_items_by_user.get(user_id.item())
            if seen is not None and seen.numel() > 0:
                all_neg_items[seen] = False
            all_neg_items[pos_items] = False
            if postfilter_global_items is not None and postfilter_global_items.numel() > 0:
                all_neg_items[postfilter_global_items] = False
            if postfilter_users is not None:
                uid = user_id.item()
                lo = int(torch.searchsorted(postfilter_users, torch.tensor(uid, device=device)))
                hi = int(
                    torch.searchsorted(
                        postfilter_users, torch.tensor(uid + 1, device=device)
                    )
                )
                items = postfilter_items[lo:hi]
                if items.numel() > 0:
                    all_neg_items[items] = False
            neg_scores = user_scores[all_neg_items]
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                auc_val = (pos_scores.unsqueeze(1) > neg_scores).float().mean()
                auc_scores.append(auc_val.item())
            else:
                auc_scores.append(0.5) # Default value if no pos or neg scores

    safe_mean = lambda x: np.mean(x) if len(x) > 0 else 0.0
    metrics = {
        "recall@k": safe_mean(recalls),
        "ndcg@k": safe_mean(ndcgs),
        "precision@k": safe_mean(precisions),
        "f1@k": safe_mean(f1_scores),
        "auc": safe_mean(auc_scores),
        "hr@k": safe_mean(hit_rates),
    }
    return metrics


def print_metrics(metrics, prefix="", k=10):
    """Prints model evaluation metrics."""
    logger.info(
        f"{prefix} Metrics: "
        f"Recall@{k}:{metrics['recall@k']:.6f}, "
        f"NDCG@{k}:{metrics['ndcg@k']:.6f}, "
        f"Precision@{k}:{metrics['precision@k']:.6f}, "
        f"HR@{k}:{metrics['hr@k']:.6f}, "
        f"AUC:{metrics['auc']:.6f}"
    )
