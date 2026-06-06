import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

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

    if is_gnn_model:
        graph_edge_index = getattr(
            data, "train_edge_index_after_remove", data.train_edge_index
        )
        user_emb, item_emb = model(edge_index=graph_edge_index.to(device))
    elif is_neumf:
        user_emb, item_emb = None, None # NeuMF scores all candidates, no pre-computed embeddings needed
    else:
        user_emb, item_emb = model.get_embeddings()

    # Create a mask for training interactions to exclude them from recommendations
    train_mask = torch.zeros(
        (data.num_users, data.num_items), dtype=torch.bool, device=device
    )
    if user_train_items is not None:
        for user_id, item_ids in user_train_items.items():
            if 0 <= user_id < data.num_users and item_ids:
                item_tensor = torch.tensor(
                    list(item_ids), dtype=torch.long, device=device
                )
                item_tensor = item_tensor[
                    (0 <= item_tensor) & (item_tensor < data.num_items)
                ]
                train_mask[user_id, item_tensor] = True
    elif hasattr(data, 'train_ratings') and not data.train_ratings.empty:
        train_pairs = torch.tensor(data.train_ratings[['user_idx', 'item_idx']].values, device=device)
        train_mask[train_pairs[:, 0], train_pairs[:, 1]] = True

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

        # Apply training mask
        scores_batch.masked_fill_(train_mask[batch_user_ids], float("-inf"))

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
            all_neg_items = torch.ones(data.num_items, dtype=torch.bool, device=device)
            all_neg_items[train_mask[user_id]] = False
            all_neg_items[pos_items] = False
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
